"""
Synthetic recovery checks for the SWIRL decision-scale toggles (swirl_scales). Each scenario simulates data in
which ONE layer is real, fits the baseline and the candidate variants on train sequences, and scores held-out
sequences. The matching variant should win on held-out LL and recover the planted structure; the others should
not gain much.

    PYTHONPATH=src JAX_PLATFORMS=cpu NATURENET_JAX_PLATFORM=cpu \
        python -m naturenet.models.irl.synth_swirl_scales --out /tmp/scales_synth [--scenario order durations ...]

Scenarios
  order       4x4 grid world; 3-gram (a, b, a) rewards carry a backtracking penalty (strong on intention 0), so
              the action at x_t depends on x_{t-1} (order-3 memory); fits N = 1..4 -> expect ~no gain at N = 2
              (same action memory as N = 1), a clear gain at 3, none beyond (N* = 3)
  durations   dwell times ~ negative binomial (mean 10, shape 6: non-geometric); fits off / logistic /
              nonparametric -> expect a rising hazard and a held-out gain
  regimes     two slow regimes (mean 150 steps) with fast (p = 0.3) vs slow (p = 0.03) intention switching;
              fits R = 1, 2 -> expect regime recovery
  history     switching probability depends on x_{t-2} (0.5 in trigger states, else 0.02); fits off / lags /
              decay -> expect a gain for lags with the lag-1 weights on the trigger states
Outputs: synth_results.csv, synth_<scenario>.png
"""
import argparse
import itertools
import os
import time

import numpy as np
import jax.numpy as jnp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from naturenet.models.irl import swirl_scales as sc
from naturenet.models.irl.swirl_multiagent import sticky_init


def rand_mdp(S, A, seed, branch=2, conc=0.5):
    rng = np.random.default_rng(seed)
    T = np.zeros((S, A, S))
    for s in range(S):
        for a in range(A):
            nx = rng.choice(S, size=branch, replace=False)
            T[s, a, nx] = rng.dirichlet(np.full(branch, conc))
    return T


def grid_mdp(nx, ny, slip=0.1):
    """Grid world, actions stay/N/S/E/W; the intended move succeeds with prob 1 - slip, else stay."""
    S = nx * ny
    T = np.zeros((S, 5, S))
    moves = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]
    for s in range(S):
        i, j = divmod(s, ny)
        for a, (di, dj) in enumerate(moves):
            i2, j2 = min(max(i + di, 0), nx - 1), min(max(j + dj, 0), ny - 1)
            s2 = i2 * ny + j2
            T[s, a, s2] += 1 - slip
            T[s, a, s] += slip
    return T


class Sim:
    """Simulator: policies from a true tree / reward; switch_fn(state dict) -> next z."""

    def __init__(self, T, K, theta, order, seed):
        self.T, self.K = T, K
        self.S, self.A = T.shape[:2]
        nodes = [c for k in range(1, order + 1) for c in itertools.product(range(self.S), repeat=k)]
        self.tree = sc.ContextTree(nodes, self.S, order, T)             # every k-gram, k <= order
        ta = self.tree.arrays()
        th = np.zeros((K, self.tree.H))
        th[:, :theta.shape[1]] = theta
        self.theta = th
        self.pi = np.exp(np.asarray(sc.log_policy(jnp.asarray(th), ta["nxt"], ta["prob"], ta["anc"], jnp.ones(K),
                                                  "additive", False)))

    def run(self, N, L, z_next, seed):
        rng = np.random.default_rng(seed)
        rows = np.zeros((N, L + 2), int)
        acts = np.zeros((N, L), int)
        zs, rs = np.zeros((N, L), int), np.zeros((N, L), int)
        for n in range(N):
            rows[n, 0] = rng.integers(self.S)
            rows[n, 1] = rng.choice(self.S, p=self.T[rows[n, 0], rng.integers(self.A)])
            st = dict(z=rng.integers(self.K), tau=1, r=0, t=0)
            for t in range(L):
                if t > 0:
                    st = z_next(st, rows[n, :t + 2], rng)
                h = self.tree.lookup(tuple(rows[n, max(0, t + 2 - self.tree.order): t + 2]))
                a = rng.choice(self.A, p=self.pi[st["z"], h])
                acts[n, t], zs[n, t], rs[n, t] = a, st["z"], st["r"]
                rows[n, t + 2] = rng.choice(self.S, p=self.T[rows[n, t + 1], a])
        return rows, acts, zs, rs


def _geom(p):
    def f(st, hist, rng):
        z = st["z"]
        return dict(st, z=1 - z if rng.random() < p else z)
    return f


def best_perm_acc(true, pred, K):
    return max(np.mean(np.asarray(p)[pred] == true) for p in itertools.permutations(range(K)))


def fit_variant(conf, T, rows_tr, a_tr, rows_te, a_te, K, seed, iters):
    cfg = sc.ScalesConfig.from_dict(conf)
    S = T.shape[0]
    tree = sc.ContextTree.build(rows_tr, S, T, order=cfg.order, min_count=cfg.context_min_count)
    dtr = sc.scales_data(rows_tr, a_tr, tree, cfg.history, S)
    dte = sc.scales_data(rows_te, a_te, tree, cfg.history, S)
    base = sticky_init(K, S, seed, sticky=0.9)
    R0 = np.random.default_rng(seed).normal(scale=0.5, size=(K, S))
    p = sc.init_params(cfg, tree, base, R0, dtr["phi"].shape[-1], seed=seed)
    t = time.time()
    r = sc.em_train_scales(p, cfg, tree, dtr, iters=iters, tol=1e-5)
    etr, ete = sc.scales_eval(r, tree, dtr), sc.scales_eval(r, tree, dte)
    return dict(cfg=cfg, tree=tree, r=r, dtr=dtr, dte=dte, tr=etr, te=ete, sec=time.time() - t,
                n_params=sc.n_params(r["params"], cfg, tree))


def scenario(name, args):
    S, A, K, L = 14, 5, 2, args.length
    rng = np.random.default_rng(args.seed)
    T = rand_mdp(S, A, args.seed)
    theta = rng.normal(scale=2.0, size=(K, S))
    if name == "order":
        # Grid world (actions stay/N/S/E/W) with a backtracking penalty on 3-grams (a, b, a): the action at x_t
        # then depends on x_{t-1} (directional persistence), which order 2 cannot represent and order 3 can.
        T, S = grid_mdp(4, 4, slip=0.1), 16
        theta = rng.normal(scale=1.0, size=(K, S))
        sim = Sim(T, K, theta, 3, args.seed)
        for h, c in enumerate(sim.tree.nodes):
            if len(c) == 3 and c[0] == c[2] and c[0] != c[1]:
                sim.theta[0, h] = -args.backtrack_penalty
                sim.theta[1, h] = -0.25 * args.backtrack_penalty
        ta = sim.tree.arrays()
        sim.pi = np.exp(np.asarray(sc.log_policy(jnp.asarray(sim.theta), ta["nxt"], ta["prob"], ta["anc"],
                                                 jnp.ones(K), "additive", False)))
        step, variants = _geom(0.05), [dict(order=n) for n in (1, 2, 3, 4)]
    else:
        sim = Sim(T, K, theta, 1, args.seed)
        if name == "durations":
            def step(st, hist, rng):
                if st["tau"] >= st.get("dwell", 0):
                    return dict(st, z=1 - st["z"], tau=1, dwell=1 + rng.negative_binomial(6, 6 / (6 + 9)))
                return dict(st, tau=st["tau"] + 1)
            variants = [dict(), dict(durations=dict(max_duration=25)),
                        dict(durations=dict(max_duration=25, hazard="nonparametric", smooth=5.0))]
        elif name == "regimes":
            def step(st, hist, rng):
                r = st["r"] if rng.random() > 1 / 150 else 1 - st["r"]
                z = 1 - st["z"] if rng.random() < (0.3 if r == 0 else 0.03) else st["z"]
                return dict(st, z=z, r=r)
            variants = [dict(), dict(hierarchy=dict(n_regimes=2, init_noise=1.0))]
        elif name == "history":
            trig = set(range(0, S, 2))

            def step(st, hist, rng):
                p = 0.5 if len(hist) >= 3 and hist[-3] in trig else 0.02
                return dict(st, z=1 - st["z"] if rng.random() < p else st["z"])
            variants = [dict(), dict(switch_history=dict(kind="lags", lags=2)),
                        dict(switch_history=dict(kind="decay", decay_steps=[2, 6]))]
        else:
            raise ValueError(name)
    rows, acts, zs, rs = sim.run(args.n_train + args.n_test, L, step, args.seed + 7)
    tr, te = slice(0, args.n_train), slice(args.n_train, None)
    out, fits = [], []
    for v in variants:
        f = fit_variant(v, T, rows[tr], acts[tr], rows[te], acts[te], K, args.seed, args.iters)
        fits.append(f)
        rec = dict(scenario=name, variant=f["cfg"].tag(), n_params=f["n_params"], LL_train=f["tr"]["LL"],
                   LL_test=f["te"]["LL"], z_acc_test=best_perm_acc(zs[te], f["te"]["path"], K), sec=f["sec"])
        if f["cfg"].R > 1:
            rec["r_acc_test"] = best_perm_acc(rs[te], f["te"]["gamma_r"].argmax(-1), f["cfg"].R)
        if f["cfg"].order > 1:
            sh = sc.reward_depth_share(f["r"], f["tree"])
            rec["deep_share_max"] = float(np.nanmax(sh[:, 1:].sum(1)))
        out.append(rec)
        print({k: (round(v_, 4) if isinstance(v_, float) else v_) for k, v_ in rec.items()})
    base = out[0]["LL_test"]
    for rec in out:
        rec["dLL_test_vs_base"] = rec["LL_test"] - base
    _plot(name, fits, out, sim, zs[te], rs[te], args.out)
    return out


def _plot(name, fits, out, sim, z_te, r_te, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    labs = [o["variant"] for o in out]
    ax.bar(range(len(out)), [o["dLL_test_vs_base"] for o in out], color="#4c72b0")
    ax.set_xticks(range(len(out)))
    ax.set_xticklabels(labs, rotation=20, ha="right", fontsize=8)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("held-out LL per step vs baseline")
    ax.set_title(f"{name}: held-out gain")
    ax = axes[1]
    if name == "durations":
        for f in fits[1:]:
            h, _ = sc.hazard_curves(f["r"])
            for k in range(h.shape[1]):
                ax.plot(np.arange(1, h.shape[2] + 1), h[0, k], label=f"{f['cfg'].tag()} z{k}")
        tau = np.arange(1, 26)
        from scipy.stats import nbinom
        pmf = nbinom.pmf(tau - 1, 6, 6 / 15)
        sf = 1 - np.concatenate([[0], np.cumsum(pmf)[:-1]])
        ax.plot(tau, pmf / np.maximum(sf, 1e-12), "k--", label="true hazard")
        ax.set_xlabel("steps in intention")
        ax.set_ylabel("hazard")
        ax.legend(fontsize=6)
    elif name == "regimes":
        f = fits[-1]
        sw = np.flatnonzero(np.ptp(r_te, axis=1) > 0)       # show a test sequence that contains a regime switch
        n = int(sw[0]) if sw.size else 0
        ax.plot(f["te"]["gamma_r"][n, :, 0], label="P(r = 0)")
        ax.plot(r_te[n], "k--", lw=0.8, label="true r")
        ax.set_xlabel("step")
        ax.legend(fontsize=7)
    elif name == "order":
        f = fits[2]
        cnt = np.bincount(np.asarray(f["dtr"]["ctx"]).ravel(), minlength=f["tree"].H)
        kl = sc.history_kl(f["r"], f["tree"], cnt)
        for k in range(kl.shape[0]):
            ax.plot(np.arange(1, kl.shape[1] + 1), kl[k], "o-", label=f"fitted z{k} ({f['cfg'].tag()})")
        ax.set_xlabel("context depth")
        ax.set_ylabel("mean KL(pi(.|h) || pi(.|x_t))")
        ax.legend(fontsize=7)
    elif name == "history":
        f = fits[1]
        W = np.abs(np.asarray(f["r"]["params"]["W1f"])).sum(1)
        S = sim.S
        ax.bar(np.arange(len(W)), W, color=["#c44e52" if (i % S) % 2 == 0 else "#8c8c8c" for i in range(len(W))])
        ax.set_xlabel("lag feature (lag1 states | lag2 states); red = trigger state")
        ax.set_ylabel("|W1| row sum")
    ax.set_title("recovery")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"synth_{name}.png"), dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="scales_synth")
    ap.add_argument("--scenario", nargs="*", default=["order", "durations", "regimes", "history"])
    ap.add_argument("--n_train", type=int, default=60)
    ap.add_argument("--n_test", type=int, default=30)
    ap.add_argument("--length", type=int, default=150)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backtrack_penalty", type=float, default=4.0, help="order scenario: depth-3 (a, b, a) penalty")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    import pandas as pd
    rows = []
    for s in args.scenario:
        rows += scenario(s, args)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "synth_results.csv"), index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
