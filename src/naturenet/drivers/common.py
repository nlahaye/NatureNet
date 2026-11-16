"""Shared helpers for naturenet.drivers (run YAMLs, K-fold file names, HIAVI reload, small statistics)."""
import copy
import glob
import json
import os
import re

import numpy as np
import pandas as pd
import yaml

AUG_RE = re.compile(r"__aug\d+$")


# ---------------------------------------------------------------------------- YAML / paths
def read_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def write_yaml(obj, path, header=""):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write(header)
        yaml.safe_dump(obj, f, sort_keys=False, width=140)
    return path


def pkg_dir():
    """naturenet package dir (config paths in the specs are relative to it, as in make_level_configs)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def abs_cfg(path, base=None):
    if path is None or os.path.isabs(path):
        return path
    base = base or pkg_dir()
    p = os.path.join(base, path)
    return p if os.path.exists(p) or not os.path.exists(path) else os.path.abspath(path)


def run_conf(spec):
    """spec: path | dict | {yaml: path, set: {key: value}} -> run config dict (overrides applied)."""
    if isinstance(spec, str):
        return read_yaml(abs_cfg(spec))
    spec = dict(spec)
    if "yaml" in spec:
        conf = read_yaml(abs_cfg(spec["yaml"]))
        for k, v in (spec.get("set") or {}).items():
            conf[k] = v
        return conf
    return spec


def atomic_csv(df, path, **kw):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    df.to_csv(tmp, index=kw.pop("index", False), **kw)
    os.replace(tmp, path)
    return path


def atomic_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=_json_default)
    os.replace(tmp, path)
    return path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ---------------------------------------------------------------------------- animals / years
def base_id(uid):
    return AUG_RE.sub("", str(uid))


def is_original(key):
    return AUG_RE.search(str(key)) is None


def year_of(group, regex=r"(?:^|_)((?:19|20)\d\d)(?:_|$)"):
    m = re.search(regex, str(group))
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------- data / folds
def hiavi_data(conf):
    """load_swirl_dataset + prepare_hiavi_data (aligned (N, T) s / a / mask, meta, groups, segments)."""
    from naturenet.models.irl.run_hiavi_kfold import prepare_hiavi_data
    from naturenet.models.irl.swirl_data import load_swirl_dataset
    return prepare_hiavi_data(conf, load_swirl_dataset(conf, allow_variable=True))


def folds(conf, data):
    from naturenet.models.irl.swirl_data import get_folds
    return get_folds(conf, data["groups"])


def fold_T(conf, data, train_index):
    from naturenet.models.irl.run_hiavi_kfold import fold_trans_prob
    return fold_trans_prob(conf, data, train_index)


def loop_conf(conf, K, run=0):
    """Config of one (K, run) of a get_stats_hiavi sweep (run_uid / seed as written by the sweep)."""
    from naturenet.models.irl.swirl_data import iter_loop_configs
    c = copy.deepcopy(conf)
    c["min_latent"] = min(int(c.get("min_latent", K)), int(K))
    c["max_latent"] = max(int(c.get("max_latent", K)), int(K))
    c["num_runs"] = max(int(c.get("num_runs", 1)), int(run) + 1)
    for lc in iter_loop_configs(c, n_hidden=int(K), run=int(run)):
        return lc
    raise ValueError(f"no loop config for K={K} run={run}")


def hiavi_files(conf, K, run=0, fold=None, sweep=True):
    """{model, ll, ll_test} of a HIAVI fit. sweep=True: get_stats_hiavi names (loop_<run>_latent_<K>_states);
    sweep=False: run_hiavi_kfold names for the YAML's own run_uid / n_hidden. fold=None: final (holdout) fit."""
    from naturenet.models.irl.run_hiavi_kfold import hiavi_names
    c = loop_conf(conf, K, run) if sweep else dict(conf, n_hidden=int(K))
    return hiavi_names(c, "" if fold is None else f"_KF_{int(fold)}")


def best_run(conf, K, n_folds, runs=None):
    """Run index with the best mean TRAIN LL over folds (selection never uses test data); 0 if nothing found."""
    runs = range(int(conf.get("num_runs", 1))) if runs is None else runs
    best, best_ll = 0, -np.inf
    for i in runs:
        lls = []
        for k in range(n_folds):
            f = hiavi_files(conf, K, i, k)["ll"]
            if os.path.exists(f):
                lls.append(float(np.load(f)["LL"]))
        if len(lls) == n_folds and np.mean(lls) > best_ll:
            best, best_ll = i, float(np.mean(lls))
    return best


def load_hiavi(path, T_env, conf=None, kin_cfg=None):
    """HIAVI object from a saved model npz (params + kinematic emissions when present)."""
    from naturenet.models.irl.hiavi_model import HIAVI
    from naturenet.models.irl.kinematics import KinematicEmissions, kin_conf
    if kin_cfg is None and conf is not None:
        kin_cfg = kin_conf(conf)
    z = np.load(path, allow_pickle=True)
    K = int(z["pis"].shape[0])
    m = HIAVI(K, T_env, discount=float(z["discount"]) if "discount" in z.files else 0.9, kinematics=kin_cfg)
    m.logp_init, m.logp_tr = np.asarray(z["logp_init"]), np.asarray(z["logp_tr"])
    m.pis, m.r, m.q = np.asarray(z["pis"]), np.asarray(z["r"]), np.asarray(z["q"])
    if KinematicEmissions.has_params(z):
        m.kin_model = KinematicEmissions.from_params(z)
        if m.kin_cfg is None:
            m.kin_cfg = {"step_feature": str(z["kin_step_feature"]) if "kin_step_feature" in z.files else "speed_kmh"}
    return m


def kinematics(conf, data):
    from naturenet.models.irl.run_hiavi_kfold import hiavi_kinematics
    return hiavi_kinematics(conf, data)


# ---------------------------------------------------------------------------- statistics
def js_divergence(p, q, eps=1e-12):
    p = np.asarray(p, float) + eps
    q = np.asarray(q, float) + eps
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * (p * np.log(p / m)).sum() + 0.5 * (q * np.log(q / m)).sum())


def run_lengths(x, mask=None):
    """Lengths of constant runs of each row of an (N, T) int array (mask False = padding, ends a row)."""
    x = np.asarray(x)
    out = []
    for i in range(x.shape[0]):
        r = x[i] if mask is None else x[i][np.asarray(mask[i], bool)]
        if len(r) == 0:
            continue
        cut = np.flatnonzero(np.diff(r) != 0) + 1
        out.extend(np.diff(np.concatenate([[0], cut, [len(r)]])).tolist())
    return np.asarray(out, dtype=np.int64)


def hist_pmf(x, n_bins):
    x = np.asarray(x, dtype=np.int64)
    h = np.bincount(np.clip(x, 0, n_bins - 1), minlength=n_bins).astype(float)
    return h / max(h.sum(), 1.0)


def bh(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    q = p[ok]
    o = np.argsort(q)
    r = q[o] * len(q) / (np.arange(len(q)) + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    v = np.empty_like(q)
    v[o] = np.clip(r, 0, 1)
    out[ok] = v
    return out


def paired_summary(d, n_boot=2000, seed=0):
    """Mean, bootstrap 95% CI, sign-flip permutation p (two-sided) and share > 0 of paired differences."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return dict(n=0, mean=np.nan, ci_low=np.nan, ci_high=np.nan, p_signflip=np.nan, frac_pos=np.nan)
    rng = np.random.default_rng(seed)
    bs = rng.choice(d, (n_boot, len(d)), replace=True).mean(1) if len(d) > 1 else np.repeat(d.mean(), 2)
    if len(d) <= 16:
        signs = np.array(np.meshgrid(*[[-1, 1]] * len(d))).reshape(len(d), -1).T
    else:
        signs = rng.choice([-1, 1], (20000, len(d)))
    null = (signs * d).mean(1)
    p = float((np.abs(null) >= abs(d.mean()) - 1e-15).mean())
    return dict(n=int(len(d)), mean=float(d.mean()), ci_low=float(np.percentile(bs, 2.5)),
                ci_high=float(np.percentile(bs, 97.5)), p_signflip=p, frac_pos=float((d > 0).mean()))


def glob_one(pattern):
    hits = sorted(glob.glob(pattern))
    return hits[0] if hits else None


def frame(rows):
    return pd.DataFrame(rows) if rows else pd.DataFrame()
