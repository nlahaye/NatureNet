"""Synthetic check of decision_context_similarity (series + latents inputs). python -m naturenet.drivers.test_decision_context"""
import os
import tempfile

import numpy as np
import pandas as pd

from naturenet.drivers.decision_context_similarity import run

rng = np.random.default_rng(0)
d = tempfile.mkdtemp()
os.makedirs(os.path.join(d, "ser"))
N, T = 6, 80
z = np.zeros((N, T), int)
for n in range(N):
    for t in range(1, T):
        z[n, t] = z[n, t - 1] if rng.random() < 0.9 else rng.integers(3)
perm = np.array([2, 0, 1])                                   # model B = relabelled A, 10% noise, fold-wise npz
zb = np.where(rng.random((N, T)) < 0.1, rng.integers(3, size=(N, T)), perm[z])
cls = np.array(["exploit", "explore", "ambiguous"])
for n in range(N):
    pd.DataFrame(dict(uid=f"an{n}", segment=0, seg_pos=np.arange(T) + 1, latent=z[n], latent_class=cls[z[n]],
                      p_exploit=(z[n] == 0) + 0.01 * rng.random(T), env=rng.integers(4, size=T))
                 ).to_csv(os.path.join(d, "ser", f"an{n}_time_series_pd.csv"), index=False)
for f in range(2):
    idx = np.arange(f, N, 2)
    keys = np.array([f"RUN_an{n}" for n in idx] + [f"RUN_an{idx[0]}__aug1"])
    P = np.concatenate([zb[idx], zb[idx[:1]]])
    np.savez(os.path.join(d, f"lat_KF_{f}.npz"), jax_path_vmap=P, seq_key=keys, seq_seg=np.zeros(len(keys), int),
             seq_start=np.zeros(len(keys), int))
pd.DataFrame(dict(state=[0, 1, 2], **{"class": ["explore", "exploit", "exploit"]})).to_csv(os.path.join(d, "ee.csv"))
conf = dict(out_dir=os.path.join(d, "out"), n_boot=20, n_null=10,
            models=dict(A=dict(kind="series", dir=os.path.join(d, "ser"), class_col="latent_class",
                               p_exploit_col="p_exploit"),
                        B=dict(kind="latents", npz=os.path.join(d, "lat_KF_{fold}.npz"), folds=[0, 1], df_run_uid="RUN",
                               seq_offset=1, exploit_explore=os.path.join(d, "ee.csv"))))
s = run(conf).iloc[0]
assert s["n_steps"] == N * T, s["n_steps"]
assert s["matched_agreement"] > 0.85 and s["AMI"] > 0.6 and s["AMI"] > s["AMI_null_mean"] + 0.3, s
ma = pd.read_csv(os.path.join(conf["out_dir"], "mode_alignment_A__B.csv"))
assert dict(zip(ma[ma.model == "A"]["mode"], ma[ma.model == "A"]["matched_mode"])) == {0: 2, 1: 0, 2: 1}
assert os.path.exists(os.path.join(conf["out_dir"], "per_fold.csv"))
print("OK")
