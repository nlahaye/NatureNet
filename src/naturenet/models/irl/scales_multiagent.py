"""
Decision-scale backends for the multi-agent test (run_hiavi_multiagent / run_scales_multiagent).

    multiagent.backend: swirl_scales | hiql_scales
    multiagent.scales: one swirl_scales variant (order, switch_history, durations, hierarchy, reward_*, hiql, ...)

Independent model: swirl_scales with the variant's toggles and coupling off.
    swirl_scales: SWIRL emission (run_swirl_kfold S-1 schedule: stage 1 no temps `epochs`, stage 2 temps_final
                  `epochs_2`), ARHMM / sticky init per fold as swirl_multiagent.SwirlBackend, restarts.
    hiql_scales:  HIQL emission (tabular context policies, HIAVI Lambda), init = HIAVI._init (seeded as the HIAVI
                  backend, hiavi_restarts), EM up to hiavi_max_iters. Order 1 with every toggle off is exactly
                  HIAVI (test E14); rewards / Q by IAVI on the context MDP after fitting.
Coupled model: the same variant with partner coupling (swirl_scales section 5)
    destination  Q~_t = (1 - alpha w_t) Q_t + alpha w_t c~_t       (not with durations and K = 2)
    timing       logit h += beta w_t (1 - c_t[z])                   (durations on)
fitted by n_outer x (em_train_scales, coupled_iters EM steps), warm-started from the independent fit, with the
partner context (w, c) built from the training animals' posteriors:
    ctx_source: independent (default)  partners' own-evidence posteriors (independent fit), a fixed input. The
                                        held-out context uses the same posteriors.
    ctx_source: mean_field             coupled posteriors, refreshed every outer round (as hiavi_multiagent).
                                        Partners' posteriors then absorb each other's switch times (an echo), which
                                        sharpens c but weakens it as a predictor for a new animal; on the synthetic
                                        hazard test the held-out gain fell from +0.026 to -0.002 per step over
                                        three rounds. Regimes stay per animal (each sequence has its own regime chain; alpha / beta are shared).
alpha = beta = 0 or w = 0 is exactly the independent model (test E12).

YAML (multiagent.scales; also multiagent.swirl for the SWIRL init / schedule, as swirl_multiagent):
    order, switch_history, durations, hierarchy, reward_l2, reward_tol, hiql: {...}   the variant
    coupling: {alpha0, fit_alpha, beta0, fit_beta, hazard}                           (defaults: multiagent.alpha0,
                                                                                      beta0 0, hazard = durations)
    coupled_iters: 10     hiql_tol: 1e-7 (relative, total LL)     ctx_source: independent | mean_field
"""
import contextlib
import copy
import io

import numpy as np
import numpy.random as npr
import jax.numpy as jnp

from naturenet.models.irl import swirl_scales as sc
from naturenet.models.irl.hiavi_model import HIAVI
from naturenet.models.irl.hiavi_multiagent import PartnerContext
from naturenet.models.irl.kinematics import KinematicEmissions, kin_conf
from naturenet.models.irl.run_hiavi_kfold import _seed, hiavi_settings, prepare_hiavi_data
from naturenet.models.irl.run_hiavi_multiagent import HiaviBackend
from naturenet.models.irl.swirl_data import init_rewards
from naturenet.models.irl.swirl_multiagent import SwirlBackend


def scales_ma_conf(conf, emission):
    """(variant dict, coupling kw, coupled_iters) from multiagent.scales."""
    s = copy.deepcopy(dict(conf.get("scales") or {}))
    cp = dict(s.pop("coupling", None) or {})
    cp.setdefault("alpha0", float(conf.get("alpha0", 0.1)))
    iters = int(s.pop("coupled_iters", (conf.get("swirl") or {}).get("coupled_iters", 10)))
    hiql_tol = s.pop("hiql_tol", 1e-7)
    ctx_source = s.pop("ctx_source", "independent")
    if ctx_source not in ("independent", "mean_field"):
        raise ValueError(f"multiagent.scales.ctx_source must be independent | mean_field, got {ctx_source!r}")
    cp["ctx_source"] = ctx_source
    s.pop("name", None)
    s["emission"] = emission
    s.setdefault("seed", int(conf.get("seed", 0)))
    return s, cp, iters, hiql_tol


def _np(T):
    T = T.todense() if hasattr(T, "todense") else T
    return np.asarray(T, dtype=float)


class _ScalesMixin:
    """Shared fit / score logic; subclasses provide rows, actions, _fit_one(K, T_np, tree, d, idx, seed)."""

    emission = "swirl"

    def _setup_scales(self):
        self.vconf, self.cpl_kw, self.coupled_iters, self.hiql_tol = scales_ma_conf(self.conf, self.emission)
        self.ctx_source = self.cpl_kw.pop("ctx_source")
        self.cfg_i = sc.ScalesConfig.from_dict(self.vconf)
        self.cfg_c = self.cfg_i.with_coupling(True, **self.cpl_kw)
        tag = self.cfg_i.tag()
        self.name = tag if tag.startswith(self.base_name) else f"{self.base_name}_{tag}"
        self._bundles = {}

    def _check_rows(self, s):
        if not np.array_equal(np.asarray(self.rows)[:, 1:-1], np.asarray(s)):
            raise ValueError("decision-scale rows[:, 1:-1] do not match the step metadata states (hiavi_align: swirl)")
        if self.acts.shape != np.asarray(s).shape:
            raise ValueError(f"actions {self.acts.shape} do not match x {np.asarray(s).shape}")

    # -- data bundles (cached per tree and index set) --
    def _bundle(self, tree, idx):
        idx = np.asarray(idx)
        key = (id(tree), idx.tobytes())
        if key not in self._bundles:
            self._bundles[key] = sc.scales_data(self.rows, self.acts, tree, self.cfg_i.history, self.S,
                                                kin=self.kin, idx=idx)
        return self._bundles[key]

    def _tree(self, T_np, idx):
        cfg = self.cfg_i
        src = self.rows if cfg.context_source == "all" else self.rows[np.asarray(idx)]
        return sc.ContextTree.build(src, self.S, T_np, order=cfg.order, min_count=cfg.context_min_count)

    def _ev(self, m, idx, ctx=None, want_kd=False):
        d = self._bundle(m["tree"], idx)
        if ctx is not None:
            d = sc.with_context(d, *ctx)
        return sc.scales_eval(m["r"], m["tree"], d, want_kd=want_kd)

    # -- backend API --
    def fit_independent(self, K, T_env, idx, kf_ind):
        idx = np.asarray(idx)
        T_np = _np(T_env)
        tree = self._tree(T_np, idx)
        d = self._bundle(tree, idx)
        best, best_ll = None, -np.inf
        for rs in range(int(self.n_restarts)):
            seed = int(np.asarray(_seed(self.yml_conf["seed"], kf_ind, rs).generate_state(1))[0]) % (2 ** 31 - 1)
            try:
                r = self._fit_one(K, T_np, tree, d, idx, seed, kf_ind, rs)
            except FloatingPointError as e:
                print(f"  restart {rs} failed: {e}")
                continue
            m = dict(r=r, tree=tree, cfg=self.cfg_i)
            ll = self._ev(m, idx)["LL"]
            if np.isfinite(ll) and ll > best_ll:
                best, best_ll = m, ll
        if best is None:
            raise RuntimeError(f"all {self.name} restarts failed (K={K}, fold {kf_ind})")
        best["gamma_r"] = self._ev(best, idx)["gamma_r"]
        return best, float(best_ll)

    def _coupled_init(self, m0):
        p = copy.deepcopy(m0["r"]["params"])
        cc = self.cfg_c.coupling
        p["cpl_alpha"] = np.asarray(sc._logit(cc.alpha0 / sc.ALPHA_MAX))
        if cc.hazard:
            p["hz"] = dict(p["hz"], beta=np.asarray(float(cc.beta0)))
        return p

    def fit_coupled(self, m0, T_env, idx, kern):
        idx = np.asarray(idx)
        tree = m0["tree"]
        d0 = self._bundle(tree, idx)
        gamma = self._ev(m0, idx)["gamma"]
        r0 = m0["r"]
        p, km = self._coupled_init(m0), r0.get("kin_model")
        tol = self._tol()
        r = None
        for _ in range(int(self.conf["n_outer"])):
            w, c = PartnerContext.context(kern, gamma)
            r = sc.em_train_scales(p, self.cfg_c, tree, sc.with_context(d0, w, c), temps=r0.get("temps"),
                                   kin_model=km, iters=self.coupled_iters, tol=tol)
            p, km = r["params"], r["kin_model"]
            m = dict(r=r, tree=tree, cfg=self.cfg_c)
            if self.ctx_source == "mean_field":
                gamma = self._ev(m, idx, ctx=(w, c))["gamma"]
        w, c = PartnerContext.context(kern, gamma)
        m = dict(r=r, tree=tree, cfg=self.cfg_c, m0=m0 if self.ctx_source == "independent" else None)
        m["gamma_r"] = self._ev(m, idx, ctx=(w, c))["gamma_r"]
        return m, (w, c)

    def context_posteriors(self, m, T_env, idx, w, c):
        """Posteriors that build the partner context for held-out animals (see ctx_source)."""
        if m.get("m0") is not None:
            return self._ev(m["m0"], idx)["gamma"]
        return self.posteriors(m, T_env, idx, w, c)

    def step_ll_indep(self, m0, T_env, idx):
        return self._ev(m0, idx)["step_ll"]

    def posteriors_indep(self, m0, T_env, idx):
        return self._ev(m0, idx)["gamma"]

    def posteriors(self, m, T_env, idx, w, c):
        return self._ev(m, idx, ctx=(w, c))["gamma"]

    def step_ll(self, m, T_env, idx, w, c):
        return self._ev(m, idx, ctx=(w, c))["step_ll"]

    def loglik(self, m, T_env, idx, w, c):
        return self._ev(m, idx, ctx=(w, c))["LL"]

    @staticmethod
    def alpha(m):
        return sc.coupling_alpha(m["r"]["params"], m["cfg"])

    @staticmethod
    def beta(m):
        return sc.coupling_beta(m["r"]["params"], m["cfg"])

    def extra(self, m):
        return dict(beta=self.beta(m), scales_variant=self.cfg_i.tag())

    @staticmethod
    def kin_model(m):
        return m["r"].get("kin_model")

    def arrays(self, m, tag):
        r, cfg, tree = m["r"], m["cfg"], m["tree"]
        p = r["params"]
        out = {f"log_Ps_{tag}": np.asarray(p["log_Ps"]), f"logpi0_{tag}": np.asarray(p["logpi0"]),
               f"logPR_{tag}": np.asarray(p["logPR"]), f"beta_{tag}": self.beta(m),
               f"scales_cfg_{tag}": cfg.to_json(), f"ctx_depth_{tag}": tree.depth,
               f"ctx_last_{tag}": tree.last}
        if "gamma_r" in m:
            out[f"gamma_r_{tag}"] = m["gamma_r"]
        h, pmf = sc.hazard_curves(r)
        if h is not None:
            out[f"hazard_{tag}"], out[f"dwell_pmf_{tag}"] = h, pmf
            if cfg.coupling.enabled and cfg.coupling.hazard:      # hazard with partners in another intention
                b = self.beta(m)
                lg = np.log(h) - np.log1p(-h) + b
                out[f"hazard_partner_{tag}"] = 1 / (1 + np.exp(-lg))
        out[f"history_kl_{tag}"] = sc.history_kl(r, tree)
        out.update(self._emission_arrays(m, tag))
        return out


class ScalesSwirlBackend(_ScalesMixin, SwirlBackend):
    """swirl_scales (SWIRL emission) with the decision-scale toggles, independent vs partner-coupled."""
    emission = "swirl"
    base_name = "SWIRL"

    def __init__(self, yml_conf, conf):
        SwirlBackend.__init__(self, yml_conf, conf)
        self._setup_scales()
        self.n_restarts = int(self.sw["restarts"])

    def prepare(self):
        data = SwirlBackend.prepare(self)
        self.rows = np.asarray(data["states"], dtype=np.int64)
        self.acts = np.asarray(data["a"], dtype=np.int64)
        self._check_rows(data["hiavi_s"])
        self.A = int(data["n_actions"])
        return data

    def _tol(self):
        return self.yml_conf.get("em_tol")

    def _fit_one(self, K, T_np, tree, d, idx, seed, kf_ind, rs):
        lp0, lPs, Rs, km = self._init(K, idx, seed)
        _, R2 = init_rewards(K, self.S, self.A, seed % (2 ** 31 - 1))
        p = sc.init_params(self.cfg_i, tree, (np.asarray(lp0), np.asarray(lPs), tuple(np.asarray(w) for w in Rs)),
                           np.asarray(R2), d["phi"].shape[-1], seed=seed)
        tol = self._tol()
        r = sc.em_train_scales(p, self.cfg_i, tree, d, temps=None, kin_model=km, iters=int(self.sw["epochs"]),
                               tol=tol)
        return sc.em_train_scales(r["params"], self.cfg_i, tree, d, temps=self._temps(K), kin_model=r["kin_model"],
                                  iters=int(self.sw["epochs_2"]), tol=tol)

    def _emission_arrays(self, m, tag):
        p = m["r"]["params"]
        R = np.asarray(sc.node_rewards(jnp.asarray(p["theta"]), jnp.asarray(m["tree"].anc), m["cfg"].reward_param))
        return {f"reward_{tag}": R[:, None, :m["tree"].S], f"theta_{tag}": np.asarray(p["theta"])}


class ScalesHiqlBackend(_ScalesMixin, HiaviBackend):
    """swirl_scales with the HIQL emission (order-N HIAVI) and the decision-scale toggles."""
    emission = "hiql"
    base_name = "HIQL"

    def __init__(self, yml_conf, conf):
        yml_conf = dict(yml_conf)
        if yml_conf.setdefault("hiavi_align", "swirl") != "swirl":
            raise ValueError("hiql_scales needs hiavi_align: swirl (contexts use the previous states)")
        HiaviBackend.__init__(self, yml_conf, conf)
        st = hiavi_settings(yml_conf)
        hq = dict((conf.get("scales") or {}).get("hiql") or {})
        for k_hq, k_st in (("pseudocount", "pseudocount"), ("discount", "discount"), ("iavi_tol", "iavi_tol"),
                           ("iavi_max_iters", "iavi_max_iters")):
            hq.setdefault(k_hq, st[k_st])
        conf = dict(conf, scales=dict(conf.get("scales") or {}, hiql=hq))
        self.conf = conf
        if st["mstep"] != "soft":
            raise ValueError("hiql_scales supports hiavi_mstep: soft only")
        self._setup_scales()
        self.n_restarts = int(yml_conf.get("hiavi_restarts", 10))
        self.max_iters = int(st["max_iters"])
        self.sticky = float(st["sticky"])
        self.kcfg = kin_conf(yml_conf)

    def prepare(self):
        data = HiaviBackend.prepare(self)
        if int(data["hiavi_offset"]) != 1:
            raise ValueError("hiql_scales needs hiavi_offset 1 (hiavi_align: swirl)")
        self.rows = np.asarray(data["states"], dtype=np.int64)
        self.acts = np.asarray(self.a, dtype=np.int64)
        self._check_rows(self.s)
        self.S, self.A = int(data["n_states"]), int(data["n_actions"])
        return data

    def _tol(self):
        return self.hiql_tol

    def _fit_one(self, K, T_np, tree, d, idx, seed, kf_ind, rs):
        hm = HIAVI(K, T_np, seed=_seed(self.yml_conf["seed"], kf_ind, rs), sticky=self.sticky)
        lp0, lP, lpis = hm._init()
        km = None
        if self.kin is not None and self.kcfg is not None:
            km = KinematicEmissions.from_conf(K, self.kcfg).init_quantiles(self.kin[idx], rng=hm.rng)
        p = sc.init_params(self.cfg_i, tree, (lp0, lP, None), lpis, d["phi"].shape[-1], seed=seed)
        return sc.em_train_scales(p, self.cfg_i, tree, d, kin_model=km, iters=self.max_iters, tol=self._tol())

    def _emission_arrays(self, m, tag):
        p = m["r"]["params"]
        out = {f"logpi_{tag}": np.asarray(p["logpi"])}
        rr = self._iavi(m)
        out[f"r_{tag}"], out[f"q_{tag}"] = rr["r"], rr["q"]
        return out

    def _iavi(self, m):
        """IAVI on the context MDP from the fitted policies (cheap stand-in for HIAVI's final expert counts)."""
        tree, cfg = m["tree"], m["cfg"]
        pis = np.exp(np.asarray(m["r"]["params"]["logpi"]))
        out = [sc.iavi_tree(tree, pis[k], cfg.hiql.discount, cfg.hiql.iavi_tol, cfg.hiql.iavi_max_iters,
                            cfg.hiql.eps) for k in range(pis.shape[0])]
        return dict(r=np.stack([o[0] for o in out]), q=np.stack([o[1] for o in out]))
