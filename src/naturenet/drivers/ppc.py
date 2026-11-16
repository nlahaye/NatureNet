"""
Test 4 - posterior predictive simulation and one-step predictive calibration of HIAVI / HIQL (+ kinematics).

Per fold, the K-fold model (trained without the test animals) and the fold's train-only T simulate every held-out
sequence n_sim times from its observed first state and length:

    z_0 ~ init,  z_t ~ P(z_t | z_t-1);   a_t ~ pi_z(. | s_t);   s_t+1 ~ T(. | s_t, a_t)
    step_t ~ (1 - p0_z) Gamma / LogNormal(s1_z, s2_z) + p0_z delta_0;   turn_t ~ vonMises(mu_z, kappa_z)

Observed vs simulated (posterior predictive checks, the test animals' own sequences):
  occupancy_js        JS divergence of environment-state occupancy (and the PPC p of the observed JS among sims)
  selection_rho       Spearman of per-state selection ratios (occupancy / availability under a random-action walk
                      on the same T from the same starts) observed vs simulated
  state_run_mean      mean run length in the same environment state (residence)
  action_js           JS of the action distribution
  dwell_mean          mean MAP-intention dwell (Viterbi on observed vs on simulated sequences)
  step_ks / turn_ks   two-sample KS statistics of step feature / turning angle (observed finite steps only)
  ppp_<stat>          posterior predictive p = P(sim stat >= observed stat)
One-step predictive (forward filter of the held-out sequence, p(a_t | a_<t, s_<=t, kin_<t)):
  log_score, brier, ece (top-label, 10 bins), acc; step PIT (randomised for zeros) KS vs U(0, 1) and a PIT histogram.

YAML:
  out_dir: /data/nlahaye/NatureNet/Blue_Whale_v2/drivers/ppc/
  models:
    - {label: hiql_kin_k5, yaml: config/v2/swirl/whales_hiavi_kfold_kin.yaml, K: 5, sweep: True}
  n_sim: 50
  seed: 0
  folds: null                 # all
Outputs: ppc_<label>_fold<k>.csv (one row per sim + observed), ppc_summary.csv, calib_<label>.csv, pit_<label>.png

    python -m naturenet.drivers.ppc -y config/v2/drivers/generated/whales/specs/ppc.yaml [--model hiql_kin_k5] [--fold 0]
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logsumexp

from naturenet.drivers.common import (abs_cfg, atomic_csv, best_run, fold_T, folds, hiavi_data, hiavi_files,
                                      hist_pmf, js_divergence, kinematics, load_hiavi, read_yaml, run_conf,
                                      run_lengths)


# ---------------------------------------------------------------------------- simulation
def sample_kin(km, z, rng):
    """step, turn for intention array z (any shape) from a fitted KinematicEmissions."""
    if km is None:
        return None, None
    s1, s2 = np.asarray(km.s1)[z], np.asarray(km.s2)[z]
    if km.step_dist == "gamma":
        step = rng.gamma(np.maximum(s1, 1e-6), np.maximum(s2, 1e-12))
    else:
        step = np.exp(rng.normal(s1, np.maximum(s2, 1e-12)))
    if km.zero_inflated:
        step = np.where(rng.random(z.shape) < np.asarray(km.p0)[z], 0.0, step)
    kap = np.maximum(np.asarray(km.kappa)[z], 1e-8)
    turn = rng.vonmises(np.asarray(km.mu)[z], kap)
    return step, turn


def simulate(model, T_env, s0, lengths, rng, policy="model"):
    """(N, L) s, a, z simulations; policy 'random' = uniform actions (availability walk). Padding: s = s0, a = -1."""
    N, L = len(s0), int(max(lengths))
    K, S, A = model.pis.shape
    s = np.tile(np.asarray(s0)[:, None], (1, L)).astype(np.int64)
    a = np.full((N, L), -1, np.int64)
    z = np.full((N, L), -1, np.int64)
    p_init = np.exp(model.logp_init - logsumexp(model.logp_init))
    P = np.exp(model.logp_tr - logsumexp(model.logp_tr, axis=1, keepdims=True))
    cT = np.cumsum(T_env, axis=2)
    cP = np.cumsum(P, axis=1)
    cpi = np.cumsum(model.pis, axis=2)
    zt = rng.choice(K, size=N, p=p_init)
    st = s[:, 0].copy()
    for t in range(L):
        live = t < np.asarray(lengths)
        if t > 0:
            u = rng.random(N)
            zt = np.minimum((u[:, None] > cP[zt]).sum(1), K - 1)
        if policy == "random":
            at = rng.integers(0, A, N)
        else:
            u = rng.random(N)
            at = np.minimum((u[:, None] > cpi[zt, st]).sum(1), A - 1)
        s[live, t] = st[live]
        a[live, t] = at[live]
        z[live, t] = zt[live]
        u = rng.random(N)
        row = cT[st, at]
        nxt = np.minimum((u[:, None] > row).sum(1), S - 1)
        nxt = np.where(row[:, -1] > 0, nxt, st)                # unobserved (s, a): stay
        st = np.where(live, nxt, st)
    return s, a, z


# ---------------------------------------------------------------------------- statistics
def seq_stats(s, a, mask, S, A, avail=None):
    occ = hist_pmf(s[mask], S)
    out = dict(occupancy=occ, action=hist_pmf(a[mask], A), state_run_mean=float(run_lengths(s, mask).mean()))
    if avail is not None:
        out["selection"] = occ / np.maximum(avail, 1e-12)
    return out


def ks(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if len(x) < 5 or len(y) < 5:
        return np.nan
    return float(stats.ks_2samp(x, y).statistic)


def one_step(model, s, a, kin):
    """Forward-filter one-step predictive of the action (and step PIT). Returns per-step arrays (flattened real)."""
    K, S, A = model.pis.shape
    lls = model._lls(s, a, kin)                             # (N, T, K) joint emission LL
    N, L = s.shape
    mask = np.asarray(a) >= 0
    logP = model.logp_tr - logsumexp(model.logp_tr, axis=1, keepdims=True)
    la_prev = None
    lp_true, brier, conf_, corr, pit = [], [], [], [], []
    km = model.kin_model
    for t in range(L):
        if t == 0:
            lpred = np.tile(model.logp_init - logsumexp(model.logp_init), (N, 1))
        else:
            lpred = logsumexp(la_prev[:, :, None] + logP[None], axis=1)
        w = np.exp(lpred - logsumexp(lpred, axis=1, keepdims=True))          # p(z_t | past)
        pa = np.einsum("nk,nka->na", w, model.pis[:, s[:, t], :].transpose(1, 0, 2))
        m = mask[:, t]
        at = np.maximum(a[:, t], 0)
        lp_true.append(np.log(np.maximum(pa[np.arange(N), at], 1e-300))[m])
        oh = np.eye(A)[at]
        brier.append(((pa - oh) ** 2).sum(1)[m])
        conf_.append(pa.max(1)[m])
        corr.append((pa.argmax(1) == at)[m])
        if km is not None and kin is not None:
            x = kin.step[:, t]
            ok = m & np.isfinite(x)
            if ok.any():
                pit.append(_step_pit(km, w[ok], x[ok]))
        lt = lpred + lls[:, t]
        lt = lt - logsumexp(lt, axis=1, keepdims=True)
        la_prev = np.where(m[:, None], lt, lpred if la_prev is None else la_prev)
    cat = lambda v: np.concatenate(v) if v else np.zeros(0)  # noqa: E731
    return dict(lp=cat(lp_true), brier=cat(brier), conf=cat(conf_), correct=cat(corr), pit=cat(pit))


def _step_pit(km, w, x, rng=None):
    rng = rng or np.random.default_rng(0)
    s1, s2 = np.asarray(km.s1), np.asarray(km.s2)
    xp = np.maximum(x, 1e-12)[:, None]
    if km.step_dist == "gamma":
        F = stats.gamma.cdf(xp, s1[None], scale=s2[None])
    else:
        F = stats.lognorm.cdf(xp, s2[None], scale=np.exp(s1)[None])
    p0 = np.asarray(km.p0)[None] if km.zero_inflated else np.zeros((1, len(s1)))
    Fz = p0 + (1 - p0) * F
    cdf = (w * Fz).sum(1)
    pz = (w * p0).sum(1)
    zero = x <= 0
    return np.where(zero, rng.random(len(x)) * pz, cdf)


def ece(conf, correct, n_bins=10):
    conf, correct = np.asarray(conf), np.asarray(correct, float)
    if not len(conf):
        return np.nan
    b = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    e = 0.0
    for i in range(n_bins):
        m = b == i
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)


# ---------------------------------------------------------------------------- driver
def run_model(spec, mspec, only_fold=None):
    conf = run_conf(dict(yaml=mspec["yaml"], set=mspec.get("set") or {}))
    K = int(mspec.get("K", conf.get("n_hidden", 5)))
    out = spec["out_dir"]
    os.makedirs(out, exist_ok=True)
    data = hiavi_data(conf)
    fl = folds(conf, data)
    kin = kinematics(conf, data)
    sweep = bool(mspec.get("sweep", True))
    run = best_run(conf, K, len(fl)) if sweep else 0
    n_sim = int(spec.get("n_sim", 50))
    s_all, a_all = data["hiavi_s"], data["hiavi_a"]
    S, A = int(data["n_states"]), int(data["n_actions"])
    summ, calib = [], []
    for k, (tr, te) in enumerate(fl):
        if only_fold is not None and k != int(only_fold):
            continue
        if spec.get("folds") is not None and k not in spec["folds"]:
            continue
        f = hiavi_files(conf, K, run, k, sweep=sweep)["model"]
        if not os.path.exists(f):
            print(f"ppc: missing {f}")
            continue
        T = np.asarray(fold_T(conf, data, tr))
        T = T.todense() if hasattr(T, "todense") else T
        T = np.asarray(T, float)
        m = load_hiavi(f, T, conf)
        s, a = s_all[te], a_all[te]
        mask = a >= 0
        lengths = mask.sum(1)
        kte = None if kin is None else kin[te]
        rng = np.random.default_rng(int(spec.get("seed", 0)) * 1000 + k)
        sa, aa, _ = simulate(m, T, s[:, 0], lengths, rng, policy="random")
        avail = hist_pmf(sa[aa >= 0], S)
        obs = seq_stats(s, a, mask, S, A, avail)
        z_obs = m.decode(s, a, kte)
        obs_dwell = float(run_lengths(z_obs, mask).mean())
        rows = [dict(fold=k, sim=-1, occupancy_js=0.0, selection_rho=1.0, state_run_mean=obs["state_run_mean"],
                     action_js=0.0, dwell_mean=obs_dwell, step_ks=0.0, turn_ks=0.0)]
        for r in range(n_sim):
            ss, as_, zs = simulate(m, T, s[:, 0], lengths, rng)
            ms = as_ >= 0
            st = seq_stats(ss, as_, ms, S, A, avail)
            step, turn = sample_kin(m.kin_model, np.maximum(zs, 0), rng) if m.kin_model is not None else (None, None)
            row = dict(fold=k, sim=r, occupancy_js=js_divergence(obs["occupancy"], st["occupancy"]),
                       selection_rho=float(stats.spearmanr(obs["selection"], st["selection"]).correlation),
                       state_run_mean=st["state_run_mean"], action_js=js_divergence(obs["action"], st["action"]),
                       true_dwell_mean=float(run_lengths(zs, ms).mean()))
            kin_s = None
            if step is not None and kte is not None:
                fin_s, fin_t = np.isfinite(kte.step) & mask, np.isfinite(kte.turn) & mask
                row["step_ks"] = ks(kte.step[fin_s], step[fin_s])
                row["turn_ks"] = ks(kte.turn[fin_t], turn[fin_t])
                from naturenet.models.irl.kinematics import KinFeatures
                kin_s = KinFeatures(np.where(np.isfinite(kte.step), step, np.nan),
                                    np.where(np.isfinite(kte.turn), turn, np.nan))
            row["dwell_mean"] = float(run_lengths(m.decode(ss, as_, kin_s), ms).mean())
            rows.append(row)
        df = pd.DataFrame(rows)
        atomic_csv(df, os.path.join(out, f"ppc_{mspec['label']}_fold{k}.csv"))
        sims = df[df["sim"] >= 0]
        o = df[df["sim"] < 0].iloc[0]
        srow = dict(model=mspec["label"], fold=k, n_test_seq=len(te), n_sim=n_sim)
        for c in ("occupancy_js", "action_js", "step_ks", "turn_ks"):
            if c in sims:
                srow[c] = float(sims[c].mean())
        srow["selection_rho"] = float(sims["selection_rho"].mean())
        for c in ("state_run_mean", "dwell_mean"):
            srow[f"obs_{c}"] = float(o[c])
            srow[f"sim_{c}"] = float(sims[c].mean())
            srow[f"ppp_{c}"] = float((sims[c] >= o[c]).mean())
        os_ = one_step(m, s, a, kte)
        srow.update(log_score=float(os_["lp"].mean()), brier=float(os_["brier"].mean()),
                    ece=ece(os_["conf"], os_["correct"]), acc=float(os_["correct"].mean()),
                    pit_ks=float(stats.kstest(os_["pit"], "uniform").statistic) if len(os_["pit"]) > 5 else np.nan)
        summ.append(srow)
        calib.append(pd.DataFrame(dict(fold=k, pit=os_["pit"])))
        print(f"ppc {mspec['label']} fold {k}: occupancy JS {srow['occupancy_js']:.4f}, dwell obs "
              f"{srow['obs_dwell_mean']:.2f} / sim {srow['sim_dwell_mean']:.2f}, log score {srow['log_score']:.4f}")
    if calib:
        pit = pd.concat(calib, ignore_index=True)
        atomic_csv(pit, os.path.join(out, f"calib_{mspec['label']}" + ("" if only_fold is None else
                                                                        f"_fold{only_fold}") + ".csv"))
    return pd.DataFrame(summ)


def collect(spec):
    """Merge ppc_summary_<label>[_fold<k>].csv parts into ppc_summary.csv and draw PIT histograms."""
    import glob
    out = spec["out_dir"]
    parts = sorted(glob.glob(os.path.join(out, "ppc_summary_*.csv")))
    if not parts:
        return None
    d = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    atomic_csv(d, os.path.join(out, "ppc_summary.csv"))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for m in d["model"].unique():
            fs = sorted(glob.glob(os.path.join(out, f"calib_{m}*.csv")))
            if not fs:
                continue
            pit = pd.concat([pd.read_csv(f) for f in fs])["pit"].to_numpy()
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.hist(pit, bins=20, range=(0, 1), density=True, color="0.5")
            ax.axhline(1.0, color="k", lw=0.8)
            ax.set_xlabel("step PIT")
            ax.set_title(f"{m}: one-step step-length PIT", fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(out, f"pit_{m}.png"), dpi=200)
            plt.close(fig)
    except Exception as ex:                      # plotting is optional
        print("ppc: PIT plot skipped:", ex)
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-y", "--yaml", required=True)
    ap.add_argument("--model", default=None, help="label (default: all models)")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--collect", action="store_true", help="only merge the per-model / per-fold summaries")
    a = ap.parse_args(argv)
    spec = read_yaml(abs_cfg(a.yaml))
    if not a.collect:
        for ms in spec["models"]:
            if a.model and ms["label"] != a.model:
                continue
            d = run_model(spec, ms, a.fold)
            if len(d):
                atomic_csv(d, os.path.join(spec["out_dir"], f"ppc_summary_{ms['label']}"
                                           + ("" if a.fold is None else f"_fold{a.fold}") + ".csv"))
    collect(spec)


if __name__ == "__main__":
    main()
