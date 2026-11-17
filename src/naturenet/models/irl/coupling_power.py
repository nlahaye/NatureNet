"""
Utility / power analysis of partner coupling on the REAL step metadata of a multi-agent config.

Question: with this species' concurrency (who is near whom, when) and sequence lengths, how large must a partner
effect on intention switching be before the leave-one-animal-out multi-agent test detects it? And does a coupling
form (mixture / align / matrix) recover it?

    python -m naturenet.models.irl.coupling_power -y <multiagent yaml> --K 5 --variant kin \
        --sim-form align --fit-forms mixture align --effects 0 0.5 1 2 3 --reps 2 [--max-heldout 6] \
        [--n-null 0] [--out <out_dir>/coupling_power]

Procedure (backend hiavi, any species / use case with a run_hiavi_multiagent YAML):
 1. data, step metadata and the partner kernel of --variant exactly as run_hiavi_multiagent; the concurrency
    report (steps with partner context per animal) is written first (ma_concurrency.csv), since it bounds power;
 2. one independent HIAVI on all animals (K intentions, policies pi_k(a | s), switching Lambda);
 3. per effect size e and replicate: intention paths are simulated for all animals jointly under
        P_t(j -> k) = softmax_k(log Lambda_jk + w_t e (B0 c_t)[k])          (sim-form align: B0 = I;
        matrix: B0 = --sim-B or I - 1/K; mixture: (1 - e w_t) Lambda + e w_t c_t)
    with the partner context c_t recomputed from the simulated partners' intentions (--sweeps Gibbs-like sweeps
    over animals, so partners are mutually consistent), and actions a_t ~ pi_z(. | s_t) at the real states
    (padding kept). Kinematic channels are not simulated (power is for the action + switching model);
 4. the multi-agent CV (run_cv, leave-one-animal-out; --max-heldout limits held-out animals to the ones with the
    most context) on the simulated actions for every --fit-forms; optional --n-null time-shift nulls per point.
Outputs (<out>): power_rows.csv (one row per effect x rep x form x held-out animal), power_summary.csv (pooled
gain, gain per context step, fraction of animals improved, median fitted coupling, detection = pooled gain > 0
[and p < 0.05 with nulls]), power.png, ma_concurrency.csv.
Effect 0 is the false-positive check. Real-data coupled results are interpretable only for effects at or above
the smallest effect detected here.
"""
import argparse
import copy
import json
import os

import numpy as np
import pandas as pd
from scipy.special import logsumexp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from naturenet.models.irl.hiavi_multiagent import PartnerContext, check_coupling, coupled_log_transitions, \
    loglinear_log_transitions
from naturenet.models.irl.run_hiavi_multiagent import HiaviBackend, build_kernel, concurrency_report, \
    draw_shifts, expand_inputs, ma_conf, read_conf, run_cv, step_metadata, window_days


def sim_B(form, K, sim_B=None):
    if form == "align":
        return np.eye(K)
    if form == "matrix":
        B = np.eye(K) - 1.0 / K if sim_B is None else np.asarray(sim_B, float).reshape(K, K)
        return B - B.mean(axis=0, keepdims=True)
    return None


def simulate(m0, s, a_real, pc, kern, form, effect, B0, rng, sweeps=3):
    """Joint simulation of intentions (all animals) and actions under the partner coupling; see module doc."""
    N, T = s.shape
    K = m0.K
    valid = a_real >= 0
    logp_init, logp_tr, logpis = m0.logp_init, m0.logp_tr, np.log(np.clip(m0.pis, 1e-300, None))

    def draw(lP, sel):
        z = np.zeros((len(sel), T), int)
        p0 = np.exp(logp_init - logsumexp(logp_init))
        z[:, 0] = rng.choice(K, size=len(sel), p=p0)
        for t in range(1, T):
            p = np.exp(lP[np.arange(len(sel)), t - 1, z[:, t - 1]])
            u = rng.random(len(sel))[:, None]
            z[:, t] = np.minimum((np.cumsum(p, axis=1) < u).sum(1), K - 1)
        return z

    z = draw(np.broadcast_to(logp_tr, (N, T - 1, K, K)), np.arange(N))    # sweep 0: independent
    if effect != 0:
        for _ in range(int(sweeps)):
            g = np.eye(K)[z] * valid[..., None]
            w, c = PartnerContext.context(kern, g)
            lP = (loglinear_log_transitions(logp_tr, effect * B0, w, c) if form != "mixture"
                  else coupled_log_transitions(logp_tr, float(effect), w, c))
            z = draw(lP, np.arange(N))
    a = np.full_like(a_real, -1)
    for n in range(N):
        for t in np.flatnonzero(valid[n]):
            p = np.exp(logpis[z[n, t], s[n, t]])
            a[n, t] = rng.choice(len(p), p=p / p.sum())
    return a, z


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-y", "--yaml", required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--variant", default=None, help="kernel variant of multiagent.variants (default: sync_variant)")
    ap.add_argument("--sim-form", default="align", choices=["mixture", "align", "matrix"])
    ap.add_argument("--sim-B", type=float, nargs="*", default=None, help="K*K values (row-major) for sim-form matrix")
    ap.add_argument("--fit-forms", nargs="+", default=["mixture", "align"])
    ap.add_argument("--effects", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0, 3.0],
                    help="beta (align / matrix scale) or alpha (mixture, in [0, 1))")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--sweeps", type=int, default=3)
    ap.add_argument("--max-heldout", type=int, default=6, help="held-out animals (most context first); 0 = all")
    ap.add_argument("--n-null", type=int, default=0, help="time-shift null replicates per point (slow)")
    ap.add_argument("--restarts", type=int, default=None, help="override hiavi_restarts (default: YAML)")
    ap.add_argument("--coupling-l2", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    yml = expand_inputs(read_conf(a.yaml))
    if a.restarts is not None:
        yml["hiavi_restarts"] = int(a.restarts)
    yml.pop("kinematics", None)                       # power of the action + switching model (see doc)
    conf = ma_conf(yml)
    vname = a.variant or conf["sync_variant"]
    if vname not in conf["variants"]:
        raise SystemExit(f"--variant {vname!r} not in multiagent.variants ({sorted(conf['variants'])})")
    base_v = {k: v for k, v in conf["variants"][vname].items() if k not in ("coupling", "coupling_l2")}
    out = a.out or os.path.join(yml["out_dir"], "coupling_power")
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    bk = HiaviBackend(yml, conf)
    data = bk.prepare()
    bk.kin = None
    pc = step_metadata(yml, data)
    conc = concurrency_report(pc, {vname: base_v}, out)
    cv_ = conc[conc["variant"] == vname].sort_values("steps_with_context", ascending=False)
    print(f"steps with partner context ({vname}): total {int(cv_['steps_with_context'].sum())}, "
          f"animals with >= {conf['min_context_steps']}: {int((cv_['steps_with_context'] >= conf['min_context_steps']).sum())}")
    if a.max_heldout:
        conf["heldout_animals"] = [str(x) for x in cv_[cv_["steps_with_context"] > 0]["animal"].head(a.max_heldout)]
        print("held-out animals:", conf["heldout_animals"])

    allidx = np.arange(pc.N)
    s, a_real = bk.s, np.array(bk.a)
    T_all = bk.fold_env(allidx)
    m0, _ = bk.fit_independent(a.K, T_all, allidx, 0)
    kern = build_kernel(pc, allidx, allidx, base_v)
    B0 = sim_B(a.sim_form, a.K, a.sim_B)
    win = window_days(conf, pc)
    animals = sorted(np.unique(pc.animal))

    rows = []
    for e in a.effects:
        for rep in range(a.reps):
            a_sim, _ = simulate(m0, s, a_real, pc, kern, a.sim_form, e, B0, rng, a.sweeps)
            bk_s = copy.copy(bk)
            bk_s.a = a_sim
            bk_s.data = dict(bk.data, hiavi_a=a_sim)
            fits = {f"{vname}__{f}": dict(base_v, coupling=check_coupling(f), coupling_l2=a.coupling_l2)
                    for f in a.fit_forms}
            cache = {}
            for r in run_cv(bk_s, conf, pc, a.K, fits, base_cache=cache, verbose=False):
                rows.append(dict(r, effect=e, rep=rep, sim_form=a.sim_form, null_id=-1))
            for nr in range(a.n_null):
                sh = draw_shifts(animals, rng, win, conf["null_min_shift_days"])
                for r in run_cv(bk_s, conf, pc, a.K, fits, times=pc.shifted_times(sh, win), null_id=nr,
                                base_cache=cache, verbose=False):
                    rows.append(dict(r, effect=e, rep=rep, sim_form=a.sim_form))
            print(f"effect {e} rep {rep} done")
            pd.DataFrame(rows).drop(columns=["d_by_state"], errors="ignore").to_csv(
                os.path.join(out, "power_rows.csv"), index=False)

    df = pd.DataFrame(rows).drop(columns=["d_by_state"], errors="ignore")
    summ = []
    for (e, rep, v), g in df.groupby(["effect", "rep", "variant"]):
        ob = g[g["null_id"] < 0]
        pooled = float(ob["d_total"].sum())
        ent = dict(effect=e, rep=rep, fit=v.split("__")[-1], pooled_gain=pooled,
                   gain_per_context_step=float(ob["d_conc"].sum() / max(ob["n_conc_steps"].sum(), 1)),
                   frac_improved=float((ob["d_total"] > 0).mean()),
                   alpha_median=float(ob["alpha"].median()),
                   align_median=float(ob["align"].median()) if "align" in ob else np.nan,
                   n_context_steps=int(ob["n_conc_steps"].sum()))
        nu = g[g["null_id"] >= 0]
        if len(nu):
            nulls = nu.groupby("null_id")["d_total"].sum().to_numpy()
            ent["p_gain"] = float((1 + np.sum(nulls >= pooled)) / (1 + len(nulls)))
        ent["detected"] = bool(pooled > 0 and ent.get("p_gain", 0.0) < 0.05)
        summ.append(ent)
    sm = pd.DataFrame(summ)
    sm.to_csv(os.path.join(out, "power_summary.csv"), index=False)
    agg = sm.groupby(["fit", "effect"]).agg(detect_rate=("detected", "mean"), pooled_gain=("pooled_gain", "mean"),
                                            align=("align_median", "mean"), alpha=("alpha_median", "mean"))
    print(agg.to_string())
    with open(os.path.join(out, "power_meta.json"), "w") as f:
        json.dump(dict(yaml=a.yaml, K=a.K, variant=vname, sim_form=a.sim_form, effects=a.effects, reps=a.reps,
                       fit_forms=a.fit_forms, n_null=a.n_null, heldout=conf.get("heldout_animals"),
                       context_steps_total=int(cv_["steps_with_context"].sum())), f, indent=1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for j, (fit, g) in enumerate(sm.groupby("fit")):
        q = g.groupby("effect")
        axes[0].errorbar(q["pooled_gain"].mean().index, q["pooled_gain"].mean(), yerr=q["pooled_gain"].std(),
                         marker="o", capsize=3, label=fit, color=f"C{j}")
        col = "alpha_median" if fit == "mixture" else "align_median"
        axes[1].plot(q[col].mean().index, q[col].mean(), marker="o", label=f"{fit} ({col.split('_')[0]})",
                     color=f"C{j}")
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_xlabel(f"simulated effect ({a.sim_form})")
    axes[0].set_ylabel("pooled held-out LL gain (nats)")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(sorted(a.effects), sorted(a.effects), "k:", lw=0.8, label="truth")
    axes[1].set_xlabel(f"simulated effect ({a.sim_form})")
    axes[1].set_ylabel("fitted coupling (CV median)")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Coupling power, K={a.K}, kernel {vname}, {int(cv_['steps_with_context'].sum())} context steps",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "power.png"), dpi=180, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
