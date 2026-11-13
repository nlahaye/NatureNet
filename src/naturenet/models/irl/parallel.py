"""
Small helpers for running independent pieces of one job in parallel, and for files that several jobs update.

cpu_map(fn, items, n_jobs)   ordered results of fn(item) (joblib / loky processes, BLAS / OpenMP threads capped
                             at 1 per worker; large numpy arguments are memory-mapped, not copied). n_jobs <= 1 runs
                             serially in this process (identical results; the default everywhere). n_jobs = -1:
                             all cores. fn must be a module-level function.
jobs_from(conf, key)         n_jobs from a YAML key, overridden by env NATURENET_<KEY> (e.g. NATURENET_HIAVI_RESTART_JOBS)
locked(path)                 exclusive advisory lock (fcntl, <path>.lock) around a read-modify-write of a shared file
update_csv(path, merge)      locked read -> merge(old DataFrame or None) -> atomic write
"""
import contextlib
import os

import pandas as pd


def jobs_from(conf, key, default=1):
    env = os.environ.get("NATURENET_" + key.upper())
    v = env if env not in (None, "") else (conf or {}).get(key, default)
    v = default if v is None else int(v)
    return (os.cpu_count() or 1) if v < 0 else v


def cpu_map(fn, items, n_jobs=1):
    items = list(items)
    if n_jobs is None or int(n_jobs) <= 1 or len(items) <= 1:
        return [fn(*it) if isinstance(it, tuple) else fn(it) for it in items]
    from joblib import Parallel, delayed, parallel_config
    n = min(int(n_jobs), len(items))
    with parallel_config(backend="loky", inner_max_num_threads=1):
        return Parallel(n_jobs=n)(delayed(fn)(*it) if isinstance(it, tuple) else delayed(fn)(it) for it in items)


@contextlib.contextmanager
def locked(path):
    import fcntl
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    with open(path + ".lock", "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def atomic_to_csv(df, path, **kw):
    tmp = f"{path}.tmp{os.getpid()}"
    df.to_csv(tmp, **kw)
    os.replace(tmp, path)


def update_csv(path, merge, index=False):
    """merge(old_df_or_None) -> new DataFrame, written atomically while holding the lock. Returns the new frame."""
    with locked(path):
        old = pd.read_csv(path) if os.path.exists(path) else None
        new = merge(old)
        atomic_to_csv(new, path, index=index)
    return new
