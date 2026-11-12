"""Gap filling: short temporal gaps first, then spatial push-pull (pyramid) fill.

Both are fully vectorised over (time, channel) so a block of days x ~100 channels is
filled in a few array passes, replacing the old per-pixel bilinear loop.
"""
import numpy as np


def temporal_fill(a, max_gap=3):
    """Linear interpolation along axis 0 for runs of <= `max_gap` missing steps.

    a : (T, ...) float32 with NaN gaps. Returns a new array. Leading/trailing gaps
    are left for the spatial fill (no extrapolation).
    """
    if max_gap <= 0 or a.shape[0] < 3:
        return a
    T = a.shape[0]
    valid = np.isfinite(a)
    part = valid.any(0) & ~valid.all(0)  # only pixels with both valid and missing steps
    if not part.any():
        return a
    if part.mean() < 0.5:
        out = a.copy()
        out[:, part] = _temporal_fill_dense(a[:, part], valid[:, part], max_gap)
        return out
    return _temporal_fill_dense(a, valid, max_gap)


def _temporal_fill_dense(a, valid, max_gap):
    T = a.shape[0]
    idx = np.arange(T).reshape((T,) + (1,) * (a.ndim - 1))
    prev = np.where(valid, idx, -1)
    np.maximum.accumulate(prev, axis=0, out=prev)
    nxt = np.where(valid, idx, T)
    nxt = np.flip(np.minimum.accumulate(np.flip(nxt, 0), axis=0), 0)
    gap = nxt - prev - 1
    fill = (~valid) & (prev >= 0) & (nxt < T) & (gap <= max_gap)
    if not fill.any():
        return a
    p = np.clip(prev, 0, T - 1)
    n = np.clip(nxt, 0, T - 1)
    vp = np.take_along_axis(a, p, axis=0)
    vn = np.take_along_axis(a, n, axis=0)
    w = (idx - prev) / np.maximum(nxt - prev, 1)
    out = a.copy()
    out[fill] = (vp + (vn - vp) * w)[fill]
    return out


def _down(v, m):
    """NaN-aware 2x2 mean with edge padding. v, m: (N, H, W)."""
    N, H, W = v.shape
    ph, pw = H % 2, W % 2
    if ph or pw:
        v = np.pad(v, ((0, 0), (0, ph), (0, pw)), mode="edge")
        m = np.pad(m, ((0, 0), (0, ph), (0, pw)), mode="edge")
    h2, w2 = v.shape[1] // 2, v.shape[2] // 2
    s = (v * m).reshape(N, h2, 2, w2, 2).sum((2, 4))
    c = m.reshape(N, h2, 2, w2, 2).sum((2, 4))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(c > 0, s / c, 0.0)
    return out.astype(np.float32), (c > 0).astype(np.float32)


def _up(v, shape):
    """Bilinear-ish 2x upsample (nearest repeat + 3x3 smoothing) cropped to `shape`."""
    u = np.repeat(np.repeat(v, 2, axis=1), 2, axis=2)[:, :shape[0], :shape[1]]
    return _smooth(u)


def _smooth(u):
    p = np.pad(u, ((0, 0), (1, 1), (1, 1)), mode="edge")
    return (4 * u + 2 * (p[:, :-2, 1:-1] + p[:, 2:, 1:-1] + p[:, 1:-1, :-2] + p[:, 1:-1, 2:])
            + p[:, :-2, :-2] + p[:, :-2, 2:] + p[:, 2:, :-2] + p[:, 2:, 2:]) / 16.0


def pushpull_fill(a, smooth_iters=4):
    """Fill every NaN in each (H, W) slice of `a` (N, H, W) from a NaN-aware pyramid.

    Valid pixels are untouched. Filled pixels are then relaxed with a few Jacobi
    (Laplace) passes so they blend smoothly into the observed field. Slices with no
    valid pixel stay NaN (the caller flags them).
    """
    a = np.asarray(a, np.float32)
    shp = a.shape
    a = a.reshape((-1,) + shp[-2:])
    m = np.isfinite(a).astype(np.float32)
    if m.all():
        return a.reshape(shp)
    v = np.where(m > 0, a, 0.0).astype(np.float32)
    levels = [(v, m)]
    while min(levels[-1][0].shape[1:]) > 1 and not levels[-1][1].all():
        levels.append(_down(*levels[-1]))
    est, em = levels[-1]
    for v_l, m_l in reversed(levels[:-1]):
        num = _up(est * em, v_l.shape[1:])
        upm = _up(em, v_l.shape[1:])
        with np.errstate(invalid="ignore", divide="ignore"):
            up = np.where(upm > 0, num / np.maximum(upm, 1e-12), 0.0)
        est = np.where(m_l > 0, v_l, up).astype(np.float32)
        em = np.maximum(m_l, (upm > 0).astype(np.float32))
    holes = m == 0
    for _ in range(int(smooth_iters)):
        p = np.pad(est, ((0, 0), (1, 1), (1, 1)), mode="edge")
        lap = 0.25 * (p[:, :-2, 1:-1] + p[:, 2:, 1:-1] + p[:, 1:-1, :-2] + p[:, 1:-1, 2:])
        est = np.where(holes, lap, est)
    empty = m.reshape(m.shape[0], -1).max(1) == 0
    est[empty] = np.nan
    return est.reshape(shp)


def missing_fraction(a, mask):
    """Fraction of water pixels that are NaN, per leading index. a: (..., H, W)."""
    n = max(int(mask.sum()), 1)
    return (~np.isfinite(a) & mask).reshape(a.shape[:-2] + (-1,)).sum(-1) / n
