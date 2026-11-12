"""Config-driven derived features computed on the filled (T, H, W) daily fields.

Each entry in `features:` is {op, name?, ...}. Ops:
  log10        var                         log10(max(x, floor))           (CHL, KD490, plankton)
  grad_mag     var                         |grad x| per km                (SST/CHL/SSS fronts)
  speed        u, v                        sqrt(u^2+v^2)
  vorticity    u, v                        relative vorticity / f         (Rossby number)
  okubo_weiss  u, v                        (s_n^2 + s_s^2 - w^2) / f^2    (eddy core < 0)
  eke          u, v                        0.5 (u'^2 + v'^2)              (use ugosa/vgosa)
  ekman_pump   curl  (N m^-3)              curl / (rho0 f), m/day         (upwelling)
  diff         a, b                        a - b                          (stratification)
  rolling_mean var, days                   trailing mean over `days`      (needs halo_back)
  anomaly      var, days                   x - trailing mean
"""
import numpy as np

from .grid import KM_PER_DEG

RHO0 = 1025.0


def _grad(x, grid):
    """d/dy, d/dx per km for (T, H, W); north-up rows (y decreases with row)."""
    gy, gx = np.gradient(x, axis=(-2, -1))
    dy = -grid.dy_km()
    dx = grid.dx_km()[:, None]
    return gy / dy, gx / dx


def _f(grid, min_lat=5.0):
    lat = np.where(np.abs(grid.lat) < min_lat, np.sign(grid.lat + 1e-9) * min_lat, grid.lat)
    from .grid import OMEGA
    return (2 * OMEGA * np.sin(np.deg2rad(lat)))[:, None]


def _rolling(x, days):
    """Trailing mean along axis 0 including the current day; first days use what exists."""
    c = np.cumsum(np.nan_to_num(x, nan=0.0), axis=0, dtype=np.float64)
    n = np.cumsum(np.isfinite(x), axis=0)
    c = np.concatenate([np.zeros_like(c[:1]), c], 0)
    n = np.concatenate([np.zeros_like(n[:1]), n], 0)
    T = x.shape[0]
    hi = np.arange(1, T + 1)
    lo = np.maximum(hi - int(days), 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return ((c[hi] - c[lo]) / (n[hi] - n[lo])).astype(np.float32)


def required_inputs(spec):
    return [spec[k] for k in ("var", "u", "v", "curl", "a", "b") if k in spec]


def default_name(spec):
    op = spec["op"]
    if "name" in spec:
        return spec["name"]
    if op in ("log10", "grad_mag"):
        return f"{op}_{spec['var']}"
    if op in ("rolling_mean", "anomaly"):
        return f"{spec['var']}_{'mean' if op == 'rolling_mean' else 'anom'}{spec['days']}d"
    if op == "diff":
        return f"{spec['a']}_minus_{spec['b']}"
    if op == "ekman_pump":
        return "ekman_pumping"
    return f"{op}_{spec.get('u', '')}"


def compute_features(fields, specs, grid):
    """fields: dict name -> (T, H, W) float32 (mutated: features are added in order)."""
    for spec in specs or []:
        miss = [k for k in required_inputs(spec) if k not in fields]
        if miss:
            if spec.get("optional", True):
                continue
            raise KeyError(f"feature {spec}: missing inputs {miss}")
        op, name = spec["op"], default_name(spec)
        if op == "log10":
            out = np.log10(np.maximum(fields[spec["var"]], float(spec.get("floor", 1e-4))))
        elif op == "grad_mag":
            gy, gx = _grad(fields[spec["var"]], grid)
            out = np.hypot(gx, gy)
        elif op == "speed":
            out = np.hypot(fields[spec["u"]], fields[spec["v"]])
        elif op in ("vorticity", "okubo_weiss"):
            u, v = fields[spec["u"]], fields[spec["v"]]
            uy, ux = _grad(u, grid)
            vy, vx = _grad(v, grid)
            k = 1e-3  # per km -> per m
            zeta = (vx - uy) * k
            f = _f(grid)
            if op == "vorticity":
                out = zeta / f
            else:
                sn, ss = (ux - vy) * k, (vx + uy) * k
                out = (sn ** 2 + ss ** 2 - zeta ** 2) / f ** 2
        elif op == "eke":
            out = 0.5 * (fields[spec["u"]] ** 2 + fields[spec["v"]] ** 2)
        elif op == "ekman_pump":
            out = fields[spec["curl"]] / (RHO0 * _f(grid)) * 86400.0
        elif op == "diff":
            out = fields[spec["a"]] - fields[spec["b"]]
        elif op == "rolling_mean":
            out = _rolling(fields[spec["var"]], spec["days"])
        elif op == "anomaly":
            out = fields[spec["var"]] - _rolling(fields[spec["var"]], spec["days"])
        else:
            raise ValueError(f"unknown feature op {op}")
        fields[name] = np.asarray(out, np.float32)
    return fields


def max_lookback(specs):
    return max([int(s.get("days", 1)) - 1 for s in specs or [] if s["op"] in ("rolling_mean", "anomaly")] + [0])
