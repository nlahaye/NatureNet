"""
Make PROJ find a usable proj.db (fixes "PROJ: proj_create_from_database: Cannot find proj.db" / "The EPSG code is
unknown" from rasterio / GDAL).

Cause: rasterio uses PROJ_DATA (or PROJ_LIB) whenever it is set, ahead of its own bundled data. Conda / pixi
activation scripts (or a shell profile) often set it to <env>/share/proj, which has no proj.db when rasterio / GDAL
came from pip wheels (the proj conda package is absent, or another env's path leaked in).

ensure_proj_data() picks, in order:
  1. rasterio's bundled proj_data (pip wheel): it must match the libproj compiled into that wheel;
  2. the current PROJ_DATA / PROJ_LIB, if it holds proj.db;
  3. pyproj's data dir, <sys.prefix>/share/proj, $CONDA_PREFIX/share/proj (first with proj.db).
It sets PROJ_DATA and PROJ_LIB (inherited by subprocesses / launcher jobs) and, if rasterio is already imported,
updates its search path. Call it before the first CRS use; it is cheap and idempotent.

    python -m naturenet.environment.proj_env            # report: current vars, candidates, choice
    eval "$(python -m naturenet.environment.proj_env --export)"   # fix the shell (scripts do this)
"""
import argparse
import importlib.util
import os
import sys

_DONE = None


def _has_db(d):
    return bool(d) and os.path.isfile(os.path.join(d, "proj.db"))


def _rasterio_wheel_dir():
    spec = importlib.util.find_spec("rasterio")
    if spec is None or not spec.origin:
        return None
    d = os.path.join(os.path.dirname(spec.origin), "proj_data")
    return d if _has_db(d) else None


def candidates():
    """[(label, path)] in priority order (paths may lack proj.db)."""
    out = [("rasterio wheel", _rasterio_wheel_dir())]
    for k in ("PROJ_DATA", "PROJ_LIB"):
        out.append((f"${k}", os.environ.get(k)))
    try:
        import pyproj.datadir
        out.append(("pyproj", pyproj.datadir.get_data_dir()))
    except Exception:
        pass
    out.append(("sys.prefix", os.path.join(sys.prefix, "share", "proj")))
    if os.environ.get("CONDA_PREFIX"):
        out.append(("CONDA_PREFIX", os.path.join(os.environ["CONDA_PREFIX"], "share", "proj")))
    return [(lab, p) for lab, p in out if p]


def ensure_proj_data(verbose=False):
    """Set PROJ_DATA / PROJ_LIB to a dir holding proj.db. -> chosen dir (None if none found; env left as is)."""
    global _DONE
    if _DONE is not None:
        return _DONE
    choice = next(((lab, p) for lab, p in candidates() if _has_db(p)), None)
    if choice is None:
        if verbose:
            print("proj_env: no proj.db found (install the proj package or reinstall rasterio)", file=sys.stderr)
        _DONE = ""
        return None
    lab, d = choice
    old = os.environ.get("PROJ_DATA") or os.environ.get("PROJ_LIB")
    os.environ["PROJ_DATA"] = os.environ["PROJ_LIB"] = d
    if "rasterio" in sys.modules:              # already imported: its search path was set from the old value
        try:
            from rasterio._env import set_proj_data_search_path
            set_proj_data_search_path(d)
        except Exception:
            pass
    if verbose and old != d:
        print(f"proj_env: PROJ_DATA {old!r} -> {d!r} ({lab})", file=sys.stderr)
    _DONE = d
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", action="store_true", help="print shell export lines only")
    a = ap.parse_args(argv)
    if a.export:
        d = ensure_proj_data()
        if d:
            print(f"export PROJ_DATA='{d}' PROJ_LIB='{d}'")
        return 0
    print("PROJ_DATA =", os.environ.get("PROJ_DATA"), "| PROJ_LIB =", os.environ.get("PROJ_LIB"))
    for lab, p in candidates():
        print(f"  {lab:<15} {'proj.db' if _has_db(p) else 'no proj.db':<11} {p}")
    d = ensure_proj_data()
    print("choice:", d or "none")
    if d:
        try:
            from rasterio.crs import CRS
            print("check: EPSG:4326 ->", CRS.from_epsg(4326).to_string())
        except Exception as e:      # noqa: BLE001
            print("check failed:", e)
            return 1
    return 0 if d else 1


if __name__ == "__main__":
    sys.exit(main())
