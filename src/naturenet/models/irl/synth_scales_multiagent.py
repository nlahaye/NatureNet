"""
Synthetic recovery for the multi-agent decision-scale models (swirl_scales coupling, HIQL and SWIRL emissions).

    python -m naturenet.models.irl.synth_scales_multiagent --out <dir> [--step 1 2 3] [--backend hiql swirl]
           [--groups 6] [--agents 4] [--L 120] [--n_null 3]

Groups of agents share a calendar (one step = one day; groups use disjoint calendars, so partners are the other
agents of the same group) and a group driver intention u_t (sticky Markov chain). Agent intentions follow
    alpha scenario (K = 3):  z_t ~ (1 - alpha) Lambda_r(z_{t-1}) + alpha onehot(u_t)   (r: per-agent regime chain)
    beta scenario  (K = 2):  semi-Markov, logit h(tau) = a + beta [u_t != z_{t-1}], switch to the other intention
Actions: soft-optimal policies of random per-intention state rewards on a random MDP (S states, A actions).
Steps (the manuscript order), each fitted with the matching swirl_scales toggles:
    1  alpha scenario, order 2 + switching-history lags     -> alpha
    2  alpha scenario with two per-agent regimes, R = 2      -> shared alpha
    3  beta scenario, durations (D = 10)                     -> beta (K = 2: alpha inactive)
Held-out: the last agent of every group (context from the training agents' posteriors, as the multi-agent CV;
--ctx_source independent (default) uses their independent-fit posteriors, mean_field the coupled ones);
the independent baseline is continued for the coupled fit's extra EM iterations (iteration-matched).
Criteria per (step, backend): coupled held-out gain > 0, alpha_hat (steps 1-2) / beta_hat (step 3) > 0 and
the calendar-shift null gain ~ 0 (mean over n_null shifted calendars, each agent shifted >= 20 days).
Writes <out>/synth_scales_ma.csv and prints a summary.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from naturenet.models.irl import swirl_scales as sc
from naturenet.models.irl.hiavi_model import HIAVI
from naturenet.models.irl.hiavi_multiagent import PartnerContext
from naturenet.models.irl.swirl_data import init_rewards
from naturenet.models.irl.swirl_multiagent import sticky_init

H_DAYS, MAX_DT = 0.5, 0.1


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def rand_mdp(S, A, rng, branch=2):
    T = np.zeros((S, A, S))
    for s in range(S):
        for a in range(A):
            nx = rng.choice(S, size=branch, replace=False)
            T[s, a, nx] = rng.dirichlet(np.ones(branch) * 2)
    return T


def policies(T, K, rng, scale=3.0):
    S = T.shape[0]
    tree = sc.ContextTree.build(np.zeros((1, 3), int), S, T, order=1)
    theta = rng.normal(scale=scale, size=(K, S))
    ta = tree.arrays()
    lp = sc.log_policy(theta, ta["nxt"], ta["prob"], ta["anc"], np.ones(K), "additive", False)
    return np.exp(np.asarray(lp))                                        # (K, S, A)


def driver(K, L, rng, stay=0.93):
    u = np.zeros(L, int)
    u[0] = rng.integers(K)
    for t in range(1, L):
        u[t] = u[t - 1] if rng.uniform() < stay else rng.choice([k for k in range(K) if k != u[t - 1]])
    return u


def simulate(step, G, M, L, S, A, seed, alpha=0.5, beta=2.5):
    rng = np.random.default_rng(seed)
    K = 2 if step == 3 else 3
    T = rand_mdp(S, A, rng)
    pi = policies(T, K, rng)
    lam = [0.9 * np.eye(K) + 0.1 / K, 0.55 * np.eye(K) + 0.45 / K]
    rows, acts, zs, times, animal, grp = [], [], [], [], [], []
    for g in range(G):
        u = driver(K, L, rng)
        for m in range(M):
            z = np.zeros(L, int)
            z[0] = rng.integers(K)
            r, tau = int(rng.integers(2)) if step == 2 else 0, 1
            for t in range(1, L):
                if step == 3:
                    lg = np.log(0.08 / 0.92) + beta * (u[t] != z[t - 1])
                    sw = rng.uniform() < 1 / (1 + np.exp(-lg))
                    z[t] = 1 - z[t - 1] if sw else z[t - 1]
                else:
                    if step == 2 and rng.uniform() < 0.02:
                        r = 1 - r
                    P = (1 - alpha) * lam[r][z[t - 1]] + alpha * np.eye(K)[u[t]]
                    z[t] = rng.choice(K, p=P / P.sum())
            s = np.zeros(L + 1, int)
            a = np.zeros(L, int)
            s[0] = rng.integers(S)
            for t in range(L):
                a[t] = rng.choice(A, p=pi[z[t], s[t]])
                s[t + 1] = rng.choice(S, p=T[s[t], a[t]])
            # swirl layout: rows (L + 1,), x = rows[1:-1] with actions a[1:]
            rows.append(s)
            acts.append(a[1:])
            zs.append(z[1:])
            times.append(g * 10000.0 + np.arange(1, L, dtype=float))
            animal.append(f"g{g}_a{m}")
            grp.append(g)
    rows, acts = np.asarray(rows), np.asarray(acts)
    times = np.asarray(times)
    pc = PartnerContext(times, np.zeros_like(times), np.zeros_like(times), np.asarray(animal), np.asarray(animal))
    return dict(rows=rows, a=acts, z=np.asarray(zs), T=T, K=K, S=S, A=A, pc=pc, group=np.asarray(grp),
                agent=np.arange(len(rows)) % M, M=M)


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

STEP_CONF = {1: dict(order=2, switch_history=dict(kind="lags", lags=1)),
             2: dict(hierarchy=dict(n_regimes=2, init_noise=0.5)),
             3: dict(durations=dict(max_duration=10))}


def fit_pair(sim, step, backend, n_outer=3, iters=60, c_iters=10, seed=0, n_null=3, ctx_source="independent"):
    K, S, A, T = sim["K"], sim["S"], sim["A"], sim["T"]
    te = np.flatnonzero(sim["agent"] == sim["M"] - 1)
    tr = np.flatnonzero(sim["agent"] != sim["M"] - 1)
    conf = dict(STEP_CONF[step], emission="hiql" if backend == "hiql" else "swirl", reward_l2=1.0, seed=seed)
    if backend == "hiql":
        conf["hiql"] = dict(backoff=5.0)
    cfg_i = sc.ScalesConfig.from_dict(conf)
    cfg_c = cfg_i.with_coupling(True, alpha0=0.1, beta0=0.0)
    tree = sc.ContextTree.build(sim["rows"][tr], S, T, order=cfg_i.order)
    dtr = sc.scales_data(sim["rows"], sim["a"], tree, cfg_i.history, S, idx=tr)
    dte = sc.scales_data(sim["rows"], sim["a"], tree, cfg_i.history, S, idx=te)
    if backend == "hiql":
        lp0, lP, lpis = HIAVI(K, T, seed=seed)._init()
        base, R0 = (lp0, lP, None), lpis
    else:
        base = sticky_init(K, S, seed)
        _, R0 = init_rewards(K, S, A, seed)
    p = sc.init_params(cfg_i, tree, base, R0, dtr["phi"].shape[-1], seed=seed)
    r0 = sc.em_train_scales(p, cfg_i, tree, dtr, iters=iters, tol=1e-8)
    # iteration-matched baseline: the independent model continued for the coupled fit's extra EM steps, so the
    # gain measures the coupling and not further convergence
    r0c = sc.em_train_scales(r0["params"], cfg_i, tree, dtr, iters=n_outer * c_iters, tol=1e-8)
    l0 = sc.scales_eval(r0c, tree, dte)["step_ll"]
    pc = sim["pc"]

    def coupled(times=None):
        kern_tr = pc.build(tr, tr, h_days=H_DAYS, max_dt_days=MAX_DT, times=times)
        kern_te = pc.build(te, tr, h_days=H_DAYS, max_dt_days=MAX_DT, times=times)
        pp = dict(r0["params"])
        pp["cpl_alpha"] = np.asarray(sc._logit(0.1 / sc.ALPHA_MAX))
        if cfg_c.coupling.hazard:
            pp["hz"] = dict(pp["hz"], beta=np.asarray(0.0))
        g = sc.scales_eval(r0, tree, dtr)["gamma"]                    # partners' own-evidence posteriors
        r = None
        for _ in range(n_outer):
            w, c = PartnerContext.context(kern_tr, g)
            r = sc.em_train_scales(pp, cfg_c, tree, sc.with_context(dtr, w, c), iters=c_iters, tol=1e-8)
            pp = r["params"]
            if ctx_source == "mean_field":
                g = sc.scales_eval(r, tree, sc.with_context(dtr, w, c))["gamma"]
        w_te, c_te = PartnerContext.context(kern_te, g)
        l1 = sc.scales_eval(r, tree, sc.with_context(dte, w_te, c_te))["step_ll"]
        return r, float((l1 - l0).mean())

    t0 = time.time()
    r1, gain = coupled()
    a_hat, b_hat = sc.coupling_alpha(r1["params"], cfg_c), sc.coupling_beta(r1["params"], cfg_c)
    rng = np.random.default_rng(seed + 99)
    null = []
    L = pc.T
    for _ in range(int(n_null)):
        sh = {an: float(rng.integers(20, max(21, L - 20))) for an in np.unique(pc.animal)}
        times = pc.times.copy()
        for i, an in enumerate(pc.animal):                            # circular shift inside the group calendar
            t0g = np.floor(times[i, 0] / 10000.0) * 10000.0
            times[i] = t0g + 1 + np.mod(times[i] - t0g - 1 + sh[an], L)
        null.append(coupled(times)[1])
    # posterior accuracy of the independent fit (best label permutation)
    import itertools
    g0 = sc.scales_eval(r0, tree, dte)["gamma"].argmax(-1)
    acc = max(np.mean(np.asarray(perm)[g0] == sim["z"][te]) for perm in itertools.permutations(range(K)))
    return dict(step=step, backend=backend, variant=cfg_i.tag(), ctx_source=ctx_source, K=K, alpha_hat=a_hat, beta_hat=b_hat,
                alpha_active=sc.alpha_active(cfg_c, K), gain_per_step=gain,
                null_gain_mean=float(np.mean(null)) if null else np.nan,
                null_gain_max=float(np.max(null)) if null else np.nan, z_acc_indep=float(acc),
                fit_sec=round(time.time() - t0, 1))


def judge(row):
    ok = row["gain_per_step"] > 0
    ok &= (row["beta_hat"] > 0) if row["step"] == 3 else (row["alpha_hat"] > 0.05)
    if np.isfinite(row["null_gain_max"]):
        ok &= row["null_gain_max"] < row["gain_per_step"]
    return bool(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=int, nargs="*", default=[1, 2, 3])
    ap.add_argument("--backend", nargs="*", default=["hiql", "swirl"])
    ap.add_argument("--groups", type=int, default=6)
    ap.add_argument("--agents", type=int, default=4)
    ap.add_argument("--L", type=int, default=120)
    ap.add_argument("--S", type=int, default=9)
    ap.add_argument("--A", type=int, default=5)
    ap.add_argument("--n_null", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ctx_source", default="independent", choices=["independent", "mean_field"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for step in a.step:
        sim = simulate(step, a.groups, a.agents, a.L, a.S, a.A, a.seed + step)
        for b in a.backend:
            row = fit_pair(sim, step, b, seed=a.seed, n_null=a.n_null, ctx_source=a.ctx_source)
            row["pass"] = judge(row)
            rows.append(row)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(a.out, "synth_scales_ma.csv"), index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
