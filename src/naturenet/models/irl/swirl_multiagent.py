"""
SWIRL backend for the multi-agent test (run_hiavi_multiagent / run_swirl_multiagent).

Independent model: SWIRL S-1 as in run_swirl_kfold.fit_swirl_final (naturenet stage, then the temperature
stage with temps_final), initialised from an ARHMM fitted on the same training animals (ssm fork; the
kinematic ARHMM when a `kinematics` block is present) or from a sticky prior (multiagent.swirl.init: sticky).

Coupled model: the same SWIRL with partner-coupled intention transitions (swirl_ext):
    P_t(j -> k) = (1 - alpha w_t) P^SWIRL_t(j -> k | x_t) + alpha w_t c_t[k]
fitted by mean-field coordinate ascent as in hiavi_multiagent.fit_coupled: posteriors of the training
animals -> partner context (w, c) -> n_outer x (em_train_ext with alpha fitted, coupled_iters EM steps),
warm-started from the independent fit. alpha = 0 / w = 0 is exactly the independent SWIRL.

YAML (multiagent.swirl, all optional):
    init: arhmm | sticky       arhmm_epochs: 50       restarts: 1
    epochs: <epochs or 30>     epochs_2: <epochs_2 or 20>     coupled_iters: 10
temps_final / em_tol / seed / n_states / per_fold_trans_prob are read from the top level as in run_swirl_kfold.
"""
import numpy as np
import numpy.random as npr
import jax.numpy as jnp

from naturenet.models.irl.hiavi_multiagent import PartnerContext
from naturenet.models.irl.kinematics import KinematicEmissions, kin_conf
from naturenet.models.irl.run_hiavi_kfold import _seed, prepare_hiavi_data
from naturenet.models.irl.swirl_data import init_rewards
from naturenet.models.irl.swirl_ext import em_train_ext, swirl_eval


def swirl_ma_conf(conf, yml_conf):
    s = dict(yml_conf.get("swirl") or {})          # top-level swirl block (optional) ...
    s.update(conf.get("swirl") or {})              # ... overridden by multiagent.swirl
    s.setdefault("init", "arhmm")
    s.setdefault("arhmm_epochs", 50)
    s.setdefault("restarts", 1)
    s.setdefault("epochs", int(yml_conf.get("epochs", 30)))
    s.setdefault("epochs_2", int(yml_conf.get("epochs_2", 20)))
    s.setdefault("coupled_iters", 10)
    return s


def sticky_init(K, S, seed, hidden=32, sticky=0.95):
    """log pi0, sticky log_Ps and a zero-output MLP (no state dependence) when no ARHMM is used."""
    rng = np.random.default_rng(seed)
    P = sticky * np.eye(K) + (1 - sticky) * rng.dirichlet(np.ones(K), K)
    P /= P.sum(1, keepdims=True)
    Rs = (rng.normal(size=(S, hidden)) * np.sqrt(2.0 / S), np.zeros((1, hidden)), np.zeros((hidden, K)),
          np.zeros((1, K)))
    return np.log(np.ones(K) / K), np.log(P), Rs


class SwirlBackend:
    name = "SWIRL"

    def __init__(self, yml_conf, conf):
        self.yml_conf, self.conf = dict(yml_conf), conf
        self.yml_conf["hiavi_align"] = "swirl"          # step metadata must use SWIRL's x = states[:, 1:-1]
        self.sw = swirl_ma_conf(conf, yml_conf)
        self.kcfg = kin_conf(yml_conf)

    def prepare(self):
        from naturenet.models.irl.run_swirl_kfold import prepare_data, swirl_kinematics

        data = prepare_data(self.yml_conf)
        data = prepare_hiavi_data(self.yml_conf, data)
        if tuple(data["hiavi_s"].shape) != tuple(np.asarray(data["xohs"]).shape[:2]) or data["hiavi_offset"] != 1:
            raise ValueError("SWIRL x and the step metadata alignment disagree")
        self.data = data
        self.xo, self.ao = data["xohs"], data["aohs"]
        self.S = int(data["n_states"])
        self.kin = swirl_kinematics(self.yml_conf, data, self.kcfg)
        return data

    def _k(self, idx):
        return None if self.kin is None else self.kin[idx]

    def fold_env(self, tr):
        from naturenet.models.irl.run_swirl_kfold import fold_trans_prob
        T = fold_trans_prob(self.yml_conf, self.data, tr)
        return jnp.asarray(T.todense() if hasattr(T, "todense") else T)

    def _temps(self, K):
        from naturenet.models.irl.run_swirl_kfold import _temps
        return jnp.asarray(_temps(self.yml_conf, "temps_final", K, [0.01] + [1.0] * (K - 1)))

    def _init(self, K, idx, seed):
        if self.sw["init"] == "sticky":
            lp0, lPs, Rs = sticky_init(K, self.S, seed)
            km = None
        else:
            from naturenet.models.irl.run_arhmm_kfold import arhmm_start_params, fit_arhmm_params
            from naturenet.models.irl.swirl_data import is_variable, step_mask
            import contextlib
            import io
            npr.seed(seed % (2 ** 31 - 1))
            with contextlib.redirect_stdout(io.StringIO()):
                xs = np.asarray(self.data["x"])[idx]
                if is_variable(self.data):                    # ssm takes per-row lists: drop padded steps
                    n_real = step_mask(self.data, 1, idx).sum(1)
                    rows = np.empty(len(xs), dtype=object)
                    for i in range(len(xs)):
                        rows[i] = xs[i, :n_real[i]]
                    xs = rows
                ar, _, _ = fit_arhmm_params(xs, K, self.S,
                                            epochs=int(self.sw["arhmm_epochs"]), kin=self._k(idx), kin_cfg=self.kcfg)
            lp0, lPs, Rs, km = arhmm_start_params(ar)
        if self.kin is not None and km is None:
            km = KinematicEmissions.from_conf(K, self.kcfg).init_quantiles(self._k(idx),
                                                                          rng=np.random.default_rng(seed))
        return jnp.asarray(lp0), jnp.asarray(lPs), tuple(jnp.asarray(r) for r in Rs), km

    def fit_independent(self, K, T_env, idx, kf_ind):
        idx = np.asarray(idx)
        tol = self.yml_conf.get("em_tol")
        temps = self._temps(K)
        best, best_ll = None, -np.inf
        for rs in range(int(self.sw["restarts"])):
            seed = int(np.asarray(_seed(self.yml_conf["seed"], kf_ind, rs).generate_state(1))[0]) % (2 ** 31 - 1)
            lp0, lPs, Rs, km = self._init(K, idx, seed)
            _, R2 = init_rewards(K, self.S, int(self.data["n_actions"]), seed % (2 ** 31 - 1))
            R0 = jnp.asarray(R2)[:, None]
            r = em_train_ext(lp0, lPs, Rs, R0, T_env, self.xo[idx], self.ao[idx], kin=self._k(idx), kin_model=km,
                             iters=int(self.sw["epochs"]), tol=tol)
            r = em_train_ext(jnp.asarray(r["logpi0"]), jnp.asarray(r["log_Ps"]), r["Rs"], jnp.asarray(r["rewards"]),
                             T_env, self.xo[idx], self.ao[idx], temps=temps, kin=self._k(idx),
                             kin_model=r["kin_model"], iters=int(self.sw["epochs_2"]), tol=tol)
            ll = swirl_eval(r, T_env, self.xo[idx], self.ao[idx], kin=self._k(idx), paths=False)["LL"]
            if np.isfinite(ll) and ll > best_ll:
                best, best_ll = r, ll
        if best is None:
            raise RuntimeError(f"all SWIRL restarts failed (K={K}, fold {kf_ind})")
        return best, float(best_ll)

    def _eval(self, r, T_env, idx, ctx=None, paths=False):
        return swirl_eval(r, T_env, self.xo[idx], self.ao[idx], kin=self._k(idx), ctx=ctx, paths=paths)

    def fit_coupled(self, m0, T_env, idx, kern):
        idx = np.asarray(idx)
        gamma = self._eval(m0, T_env, idx)["gamma"]
        r, alpha = m0, float(self.conf["alpha0"])
        for _ in range(int(self.conf["n_outer"])):
            w, c = PartnerContext.context(kern, gamma)
            r = em_train_ext(jnp.asarray(r["logpi0"]), jnp.asarray(r["log_Ps"]), r["Rs"], jnp.asarray(r["rewards"]),
                             T_env, self.xo[idx], self.ao[idx], temps=r.get("temps"), kin=self._k(idx),
                             kin_model=r.get("kin_model"), ctx=(w, c), alpha=alpha, fit_alpha=True,
                             iters=int(self.sw["coupled_iters"]), tol=self.yml_conf.get("em_tol"))
            alpha = r["alpha"]
            gamma = self._eval(r, T_env, idx, ctx=(w, c))["gamma"]
        w, c = PartnerContext.context(kern, gamma)
        return r, (w, c)

    def step_ll_indep(self, m0, T_env, idx):
        return self._eval(m0, T_env, idx)["step_ll"]

    def posteriors_indep(self, m0, T_env, idx):
        return self._eval(m0, T_env, idx)["gamma"]

    def posteriors(self, m, T_env, idx, w, c):
        return self._eval(m, T_env, idx, ctx=(w, c))["gamma"]

    def step_ll(self, m, T_env, idx, w, c):
        return self._eval(m, T_env, idx, ctx=(w, c))["step_ll"]

    def loglik(self, m, T_env, idx, w, c):
        return self._eval(m, T_env, idx, ctx=(w, c))["LL"]

    @staticmethod
    def alpha(m):
        return float(m.get("alpha", 0.0))

    @staticmethod
    def kin_model(m):
        return m.get("kin_model")

    @staticmethod
    def arrays(m, tag):
        return {f"log_Ps_{tag}": np.asarray(m["log_Ps"]), f"reward_{tag}": np.asarray(m["rewards"]),
                f"logpi0_{tag}": np.asarray(m["logpi0"])}
