"""
Coupled-intention (multi-agent) HIAVI: a mean-field multi-agent extension of hiavi_model.HIAVI.

Each animal i keeps the HIAVI emission model (intention-specific Boltzmann policies pi_k(a | s) over the
abstract environment states, rewards recovered by IAVI, shared by all animals so intention labels mean the
same thing across animals). The intention chain of animal i is coupled to the other animals' intentions at
the same calendar time through a "partner context":

    P_t^i(z_t = k | z_{t-1} = j) = (1 - alpha * w_t^i) * Lambda[j, k] + alpha * w_t^i * c_t^i[k]

  c_t^i  (K,)  kernel-weighted mean intention posterior of the OTHER animals near time tau_t^i
               (time kernel exp(-|dt| / h_days) within +/- max_dt_days, optionally x exp(-d / lambda_km)
               and a hard max_km cut; augmented copies of a partner are averaged, never pooled as extra
               animals; copies of animal i itself are excluded);
  w_t^i  [0,1) partner availability, 1 - exp(-mass), mass = summed kernel weight per partner animal;
               w = 0 where no other animal is tracked within the window, so the model reduces to HIAVI;
  alpha  [0,1) coupling strength (alpha = 0 is HIAVI; the models are nested).

Log-linear coupling (coupling: align | matrix; alpha is then unused): partners shift the switch logits instead
of being copied,

    P_t^i(z_t = k | z_{t-1} = j) = softmax_k( log Lambda[j, k] + w_t^i * sum_m B[k, m] c_t^i[m] )

  matrix  B (K, K) free: B[k, m] > 0 = partners in intention m make k more likely (attraction / following),
          < 0 = less likely (avoidance). Only differences between rows k matter (adding d_m to column m changes
          nothing); the ridge penalty coupling_l2 / 2 * ||B||^2 picks the column-centred solution.
  align   B = beta * I (one parameter): beta > 0 = alignment with partners, beta < 0 = anti-alignment.
B = 0 (or w = 0) is exactly HIAVI, so both forms are nested in the independent model like alpha = 0. The M-step
for (Lambda, B) maximises the concave expected complete-data log-likelihood sum xi log P (+ ridge) by L-BFGS.
Kernel option lag_days (PartnerContext.build): the context uses the partners' intentions lag_days EARLIER
(lag > 0: do partners lead?), with the same time / distance kernel around the shifted time.

alpha > 0 with a held-out gain means intention switches of concurrently tracked animals are aligned in
time beyond what each animal's own state sequence explains. It does not separate social interaction from a
shared external driver (season, tide, prey pulse); the proximity-weighted variant and the time-shift null
in run_hiavi_multiagent.py are the controls for that.

Fitting (coordinate ascent / mean-field): start from an independent HIAVI fit; repeat n_outer times
  (1) context from the current posteriors of the source (training) animals,
  (2) EM for (pi0, Lambda, alpha, policies) with the context fixed,
  (3) posteriors under the new model.
M-step for the mixture transition: responsibilities rho = alpha w c_k / P_t(j, k) split the expected
transition counts xi_t(j, k) between Lambda and the context; Lambda ∝ sum xi (1 - rho); alpha maximises the
concave sum_t [m_t log alpha + (1 - m_t) log(1 - alpha w_t)], m_t = sum_jk xi_t rho_t (1-D bounded search).
"""
import numpy as np
from scipy import sparse
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp

from naturenet.models.irl.hiavi_model import HIAVI, emission_loglik, iavi, softmax_policy, step_mask

TINY = 1e-300


# ---------------------------------------------------------------------------
# Time-varying HMM pieces
# ---------------------------------------------------------------------------

def coupled_log_transitions(logp_tr, alpha, w, c):
    """logp_tr (K, K); w (N, T); c (N, T, K) -> log P_t (N, T-1, K, K) for transitions t-1 -> t (t = 1..T-1)."""
    lam = np.exp(logp_tr)[None, None]                        # (1, 1, K, K)
    wt = (alpha * w[:, 1:])[..., None, None]                 # (N, T-1, 1, 1)
    P = (1.0 - wt) * lam + wt * c[:, 1:, None, :]            # rows j mix Lambda[j] with c_t
    return np.log(np.maximum(P, TINY))


COUPLING_FORMS = ("mixture", "align", "matrix")


def check_coupling(form):
    form = str(form or "mixture").lower()
    if form not in COUPLING_FORMS:
        raise ValueError(f"coupling must be one of {COUPLING_FORMS}, got {form!r}")
    return form


def loglinear_log_transitions(logp_tr, B, w, c):
    """logp_tr (K, K); B (K, K); w (N, T); c (N, T, K) -> log P_t (N, T-1, K, K): softmax_k(log Lambda + w B c)."""
    shift = w[:, 1:, None] * np.einsum("ntm,km->ntk", c[:, 1:], B)            # (N, T-1, K) destination shift
    lg = logp_tr[None, None] + shift[:, :, None, :]
    return lg - logsumexp(lg, axis=-1, keepdims=True)


def _log_softmax_rows(L):
    return L - logsumexp(L, axis=-1, keepdims=True)


def loglinear_mstep(logp_tr, B, xi, w, c, l2=1.0, form="matrix", pseudocount=1e-3, maxiter=200):
    """
    M-step of (Lambda, B) for the log-linear coupling. xi (N, T-1, K, K) expected transitions; w (N, T), c (N, T, K)
    (destination steps 1..T-1 are used). Steps without context (w = 0) only enter through their summed counts.
    Maximises sum xi log P - l2 / 2 * ||B||^2 (form align: B = beta I, penalty l2 / 2 * beta^2). Returns
    (logp_tr, B).
    """
    K = logp_tr.shape[0]
    wt = w[:, 1:].ravel()
    X = xi.reshape(-1, K, K)
    on = wt > 0
    C0 = X[~on].sum(0) + float(pseudocount)
    n0 = C0.sum(1)
    Xs, ws, cs = X[on], wt[on], c[:, 1:].reshape(-1, K)[on]
    ns = Xs.sum(-1)                                                              # (M, K)
    ntot = max(float(X.sum()), 1.0)
    diag = form == "align"

    def unpack(th):
        L = th[:K * K].reshape(K, K)
        Bm = th[-1] * np.eye(K) if diag else th[K * K:].reshape(K, K)
        return L, Bm

    def fg(th):
        L, Bm = unpack(th)
        lsL = _log_softmax_rows(L)
        f = float((C0 * lsL).sum())
        gL = C0 - n0[:, None] * np.exp(lsL)
        gB = np.zeros((K, K))
        if len(ws):
            b = ws[:, None] * (cs @ Bm.T)                                         # (M, K)
            lg = L[None] + b[:, None, :]
            lP = lg - logsumexp(lg, axis=-1, keepdims=True)
            f += float((Xs * lP).sum())
            R = Xs - ns[..., None] * np.exp(lP)                                   # (M, K, K)
            gL = gL + R.sum(0)
            G = R.sum(1)                                                          # (M, K): d/d shift_k
            gB = (G * ws[:, None]).T @ cs                                         # (K, K)
        if diag:
            beta = th[-1]
            f -= 0.5 * l2 * beta ** 2
            gth = np.concatenate([gL.ravel(), [np.trace(gB) - l2 * beta]])
        else:
            f -= 0.5 * l2 * float((Bm ** 2).sum())
            gth = np.concatenate([gL.ravel(), (gB - l2 * Bm).ravel()])
        return -f / ntot, -gth / ntot

    th0 = np.concatenate([np.asarray(logp_tr, float).ravel(),
                          [float(np.trace(B)) / K] if diag else np.asarray(B, float).ravel()])
    res = minimize(fg, th0, jac=True, method="L-BFGS-B", options=dict(maxiter=int(maxiter)))
    L, Bm = unpack(res.x)
    if not diag:
        Bm = Bm - Bm.mean(axis=0, keepdims=True)                                 # same P; column-centred
    return _log_softmax_rows(L), Bm


def coupling_summary(form, alpha, B):
    """Scalars for tables: alpha (mixture), align (beta, or the mean centred diagonal of B), B_fro (norm of B)."""
    B = np.asarray(B, float)
    if form == "mixture":
        return dict(coupling=form, alpha=float(alpha), align=np.nan, B_fro=np.nan)
    Bc = B - B.mean(axis=0, keepdims=True)
    return dict(coupling=form, alpha=0.0, align=float(np.mean(np.diag(Bc))) if form == "matrix" else float(B[0, 0]),
                B_fro=float(np.sqrt((Bc ** 2).sum())))


def forward_backward_tv(logp_init, logp_tr_t, lls, mask=None):
    """
    lls (N, T, K); logp_tr_t (N, T-1, K, K). Returns log gamma (N, T, K), xi (N, T-1, K, K) (per step,
    not summed), marginal log-likelihood per sequence (N,) and one-step predictive log-likelihoods (N, T)
    (log p(a_t | a_<t); they sum to the sequence log-likelihood).
    mask (N, T) bool (None = all real; padded steps of segment_policy: variable): padded steps carry alpha
    unchanged (identity transition, zero emission), so ll / real-step posteriors are exact; padded steps get
    log gamma -inf, xi 0 and step LL 0.
    """
    N, T, K = lls.shape
    la = np.empty((N, T, K))
    la[:, 0] = logp_init + lls[:, 0]
    for t in range(1, T):
        la[:, t] = logsumexp(la[:, t - 1][:, :, None] + logp_tr_t[:, t - 1], axis=1) + lls[:, t]
        if mask is not None:
            la[:, t] = np.where(mask[:, t, None], la[:, t], la[:, t - 1])
    lb = np.zeros((N, T, K))
    for t in range(T - 2, -1, -1):
        lb[:, t] = logsumexp(logp_tr_t[:, t] + (lls[:, t + 1] + lb[:, t + 1])[:, None, :], axis=2)
        if mask is not None:
            lb[:, t] = np.where(mask[:, t + 1, None], lb[:, t], 0.0)
    lZ = logsumexp(la, axis=2)                               # (N, T)
    ll = lZ[:, -1]
    step = np.diff(np.concatenate([np.zeros((N, 1)), lZ], axis=1), axis=1)
    lg = la + lb - ll[:, None, None]
    if T > 1:
        lx = la[:, :-1, :, None] + logp_tr_t + (lls[:, 1:] + lb[:, 1:])[:, :, None, :] - ll[:, None, None, None]
        xi = np.exp(lx)
        if mask is not None:
            xi = np.where(mask[:, 1:, None, None], xi, 0.0)
    else:
        xi = np.zeros((N, 0, K, K))
    if mask is not None:
        lg = np.where(mask[..., None], lg, -np.inf)
        step = np.where(mask, step, 0.0)
    return lg, xi, ll, step


def viterbi_tv(logp_init, logp_tr_t, lls, mask=None):
    """MAP intentions (N, T) with time-varying transitions; -1 on padded steps."""
    N, T, K = lls.shape
    d = logp_init + lls[:, 0]
    bp = np.zeros((N, T, K), dtype=np.int64)
    for t in range(1, T):
        sc = d[:, :, None] + logp_tr_t[:, t - 1]
        bp[:, t] = sc.argmax(axis=1)
        dn = sc.max(axis=1) + lls[:, t]
        if mask is not None:
            bp[:, t] = np.where(mask[:, t, None], bp[:, t], np.arange(K)[None])
            dn = np.where(mask[:, t, None], dn, d)
        d = dn
    z = np.zeros((N, T), dtype=np.int64)
    z[:, -1] = d.argmax(axis=1)
    for t in range(T - 1, 0, -1):
        z[:, t - 1] = bp[np.arange(N), t, z[:, t]]
    return z if mask is None else np.where(mask, z, -1)


# ---------------------------------------------------------------------------
# Partner context
# ---------------------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


class PartnerContext:
    """
    Step metadata for every sequence: times (N, T) in days, lat / lon (N, T), animal (N,) base id, key (N,).
    build(tgt, src, ...) precomputes a sparse (n_tgt_steps, n_src_steps) kernel matrix; context(gamma_src)
    then gives (w, c) for the target steps in one sparse product (cheap to redo every outer iteration).
    valid (N, T) bool: real steps (default all). Padded steps (segment_policy: variable) have NaN times, are
    never a source and get no context (w = 0).
    """

    def __init__(self, times, lat, lon, animal, key, valid=None):
        self.times = np.asarray(times, dtype=float)
        self.lat = np.asarray(lat, dtype=float)
        self.lon = np.asarray(lon, dtype=float)
        self.animal = np.asarray(animal).astype(str)
        self.key = np.asarray(key).astype(str)
        self.N, self.T = self.times.shape
        self.valid = np.ones((self.N, self.T), bool) if valid is None else np.asarray(valid, bool)
        if self.valid.shape != self.times.shape:
            raise ValueError(f"PartnerContext: valid {self.valid.shape} != times {self.times.shape}")
        if not np.all(np.isfinite(self.times[self.valid])):
            raise ValueError("PartnerContext: non-finite step times")

    def shifted_times(self, shifts=None, window=None):
        """
        Circularly shift each animal's calendar by shifts[animal] days inside window (t0, t1) (null model).
        window may also be a list of (t0, t1) seasons (e.g. 2015 and 2017 tracks): each step is shifted
        inside the season that contains it, so no track is moved into an untracked gap; steps outside every
        season are left as they are.
        """
        if not shifts:
            return self.times
        wins = _as_windows(window) or [(np.nanmin(self.times), np.nanmax(self.times) + 1e-6)]
        out = self.times.copy()
        for an, dlt in shifts.items():
            m = np.asarray(self.animal == an)
            m = m.reshape(m.shape + (1,) * (self.times.ndim - m.ndim))    # (N,) animal ids vs (N, T) step times
            for t0, t1 in wins:
                L = t1 - t0
                mw = m & (self.times >= t0) & (self.times < t1)
                out[mw] = t0 + np.mod(self.times[mw] - t0 + float(dlt), L)
        return out

    def build(self, tgt, src, h_days=1.0, max_dt_days=None, lambda_km=None, max_km=None, times=None, lag_days=0.0):
        """
        Kernel matrix for target sequences tgt and source sequences src (index arrays). Same-animal pairs are
        excluded; each partner animal's weight is divided by its number of keys (augmented copies), so copies
        act as one animal. times overrides the step times of every sequence (time-shift null).
        lag_days > 0: the kernel is centred on t - lag_days (partners' intentions lag_days earlier); the distance
        is still to the partner's position at that earlier step.
        """
        tgt, src = np.asarray(tgt), np.asarray(src)
        times = self.times if times is None else np.asarray(times, dtype=float)
        max_dt = float(max_dt_days) if max_dt_days is not None else 3.0 * float(h_days)
        st = times[src].ravel()
        s_an = np.repeat(self.animal[src], self.T)
        s_lat, s_lon = self.lat[src].ravel(), self.lon[src].ravel()
        n_keys = {a: len(set(self.key[src][self.animal[src] == a])) for a in np.unique(self.animal[src])}
        s_norm = np.array([1.0 / n_keys[a] for a in s_an])
        s_ok = np.flatnonzero(self.valid[src].ravel())                # padded source steps are never used
        order = s_ok[np.argsort(st[s_ok], kind="stable")]
        st_sorted = st[order]

        tt = times[tgt].ravel() - float(lag_days or 0.0)
        t_ok = self.valid[tgt].ravel()
        t_an = np.repeat(self.animal[tgt], self.T)
        t_lat, t_lon = self.lat[tgt].ravel(), self.lon[tgt].ravel()
        tq = np.where(t_ok, tt, 0.0)
        lo = np.searchsorted(st_sorted, tq - max_dt, side="left")
        hi = np.searchsorted(st_sorted, tq + max_dt, side="right")
        rows, cols, vals, dists, dts = [], [], [], [], []
        for r in np.nonzero((hi > lo) & t_ok)[0]:
            j = order[lo[r]:hi[r]]
            j = j[s_an[j] != t_an[r]]
            if not len(j):
                continue
            dt = np.abs(st[j] - tt[r])
            wv = np.exp(-dt / float(h_days)) * s_norm[j]
            d = haversine_km(t_lat[r], t_lon[r], s_lat[j], s_lon[j])
            if max_km is not None:
                keep = d <= float(max_km)
                j, wv, d, dt = j[keep], wv[keep], d[keep], dt[keep]
            if lambda_km is not None:
                wv = wv * np.exp(-d / float(lambda_km))
            rows.append(np.full(len(j), r))
            cols.append(j)
            vals.append(wv)
            dists.append(d)
            dts.append(dt)
        n_t, n_s = len(tt), len(st)
        if rows:
            rows, cols, vals = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
            self._last_pairs = dict(dist_km=np.concatenate(dists), dt_days=np.concatenate(dts))
        else:
            rows = cols = np.zeros(0, dtype=np.int64)
            vals = np.zeros(0)
            self._last_pairs = dict(dist_km=np.zeros(0), dt_days=np.zeros(0))
        Km = sparse.csr_matrix((vals, (rows, cols)), shape=(n_t, n_s))
        return dict(K=Km, n_tgt=len(tgt), n_src=len(src), T=self.T)

    @staticmethod
    def context(kern, gamma_src):
        """gamma_src (n_src, T, K) -> w (n_tgt, T), c (n_tgt, T, K). c = uniform where w = 0 (unused there)."""
        Km, T = kern["K"], kern["T"]
        Kd = gamma_src.shape[-1]
        g = gamma_src.reshape(-1, Kd)
        mass = np.asarray(Km.sum(axis=1)).ravel()
        num = Km @ g
        c = np.where(mass[:, None] > 0, num / np.maximum(mass[:, None], TINY), 1.0 / Kd)
        w = 1.0 - np.exp(-mass)
        return w.reshape(kern["n_tgt"], T), c.reshape(kern["n_tgt"], T, Kd)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CoupledHIAVI(HIAVI):
    """
    HIAVI with partner-coupled intention transitions (see module docstring).

    m = CoupledHIAVI(K, T_env, ...).init_from(hiavi_model)            # warm start (labels kept)
    m.fit_em(s, a, w, c)                                              # EM with fixed context
    m.loglik(s, a, w, c) / m.step_loglik(...) / m.posteriors(...) / m.decode(...)
    fit_alpha=False keeps alpha at alpha0 (alpha0 = 0 -> plain HIAVI with the same code path).
    coupling: mixture (alpha, default) | align (beta) | matrix (B); coupling_l2 = ridge on B (log-linear forms).
    """

    def __init__(self, n_latents, T_env, alpha0=0.1, fit_alpha=True, lambda_pseudocount=1e-3, coupling="mixture",
                 coupling_l2=1.0, **kw):
        super().__init__(n_latents, T_env, **kw)
        self.coupling = check_coupling(coupling)
        self.alpha = float(alpha0) if self.coupling == "mixture" else 0.0
        self.fit_alpha = bool(fit_alpha)
        self.lambda_pseudocount = float(lambda_pseudocount)
        self.coupling_l2 = float(coupling_l2)
        self.B = np.zeros((n_latents, n_latents))

    def log_trans(self, logp_tr, w, c, alpha=None, B=None):
        """log P_t (N, T-1, K, K) under the model's coupling form."""
        w, c = np.asarray(w, float), np.asarray(c, float)
        if self.coupling == "mixture":
            return coupled_log_transitions(logp_tr, self.alpha if alpha is None else alpha, w, c)
        return loglinear_log_transitions(logp_tr, self.B if B is None else B, w, c)

    def coupling_stats(self):
        return coupling_summary(self.coupling, self.alpha, self.B)

    def init_from(self, m):
        self.logp_init, self.logp_tr = np.array(m.logp_init), np.array(m.logp_tr)
        self.pis = np.array(m.pis)
        self._logpis_em = np.log(self.pis)
        self.kin_cfg = getattr(m, "kin_cfg", None)
        self.kin_model = m.kin_model.copy() if getattr(m, "kin_model", None) is not None else None
        return self

    def _alpha_mstep(self, xi, rho, w):
        if not self.fit_alpha:
            return self.alpha
        m_t = (xi * rho).sum(axis=(-1, -2)).ravel()      # expected "followed the context" per transition
        n_t = xi.sum(axis=(-1, -2)).ravel()              # = 1 per transition
        w_t = w[:, 1:].ravel()
        keep = w_t > 0
        if not np.any(keep):
            return 0.0
        m_t, n_t, w_t = m_t[keep], n_t[keep], w_t[keep]

        def nq(al):
            return -(np.sum(m_t) * np.log(max(al, TINY)) + np.sum((n_t - m_t) * np.log(np.maximum(1 - al * w_t, TINY))))

        res = minimize_scalar(nq, bounds=(0.0, 1.0 - 1e-6), method="bounded", options=dict(xatol=1e-6))
        return float(res.x)

    def fit_em(self, s, a, w, c, max_iters=None, tol=None, verbose=False, kin=None):
        s, a = np.asarray(s, dtype=np.int64), np.asarray(a, dtype=np.int64)
        w, c = np.asarray(w, dtype=float), np.asarray(c, dtype=float)
        if w.shape != s.shape or c.shape != s.shape + (self.K,):
            raise ValueError(f"w {w.shape} / c {c.shape} do not match s {s.shape} and K={self.K}")
        max_iters = self.max_iters if max_iters is None else int(max_iters)
        tol = self.tol if tol is None else float(tol)
        if not hasattr(self, "_logpis_em"):
            self.logp_init, self.logp_tr, self._logpis_em = self._init()
        logp_init, logp_tr, logpis, alpha, B = self.logp_init, self.logp_tr, self._logpis_em, self.alpha, self.B
        mask = step_mask(a)                                   # padded steps (a = -1)
        n_real = s.size if mask is None else int(mask.sum())
        hist, prev = [], -np.inf
        for it in range(max_iters):
            lP = self.log_trans(logp_tr, w, c, alpha, B)
            lg, xi, ll, _ = forward_backward_tv(logp_init, lP, self._emit(logpis, s, a, kin), mask)
            tot = float(ll.sum() / n_real)
            if not np.isfinite(tot):
                raise FloatingPointError(f"CoupledHIAVI LL is {tot} at iteration {it}")
            hist.append(tot)
            if it > 0 and abs(tot - prev) < tol:
                break
            prev = tot
            g = np.exp(lg)
            logpis = self._logpis(self._policies_from_counts(g, s, a))
            if self.kin_model is not None and kin is not None:
                self.kin_model.m_step(g, kin)
            logp_init = logsumexp(lg[:, 0], axis=0) - np.log(s.shape[0])
            if self.coupling != "mixture":           # log-linear coupling: joint (Lambda, B) M-step
                logp_tr, B = loglinear_mstep(logp_tr, B, xi, w, c, self.coupling_l2, self.coupling,
                                             self.lambda_pseudocount)
                if verbose:
                    print(f"  coupled EM {it}: LL/step {tot:.5f} {coupling_summary(self.coupling, 0, B)}")
                continue
            # mixture transition M-step
            lam = np.exp(logp_tr)[None, None]
            wt = (alpha * w[:, 1:])[..., None, None]
            ctx = wt * c[:, 1:, None, :]
            P = (1 - wt) * lam + ctx
            rho = np.where(P > 0, ctx / np.maximum(P, TINY), 0.0)
            cnt = (xi * (1 - rho)).sum(axis=(0, 1)) + self.lambda_pseudocount
            logp_tr = np.log(cnt / cnt.sum(axis=1, keepdims=True))
            alpha = self._alpha_mstep(xi, rho, w)
            if verbose:
                print(f"  coupled EM {it}: LL/step {tot:.5f} alpha {alpha:.4f}")
        self.logp_init, self.logp_tr, self._logpis_em, self.alpha, self.B = logp_init, logp_tr, logpis, alpha, B
        self.ll_history = np.asarray(hist)
        return self

    def finalize(self, s, a, w, c, kin=None):
        """IAVI once per intention from the final responsibilities (as HIAVI.fit)."""
        lg, _, _, _ = self._fb(s, a, w, c, self._logpis_em, kin)
        expert = self._policies_from_counts(np.exp(lg), np.asarray(s, dtype=np.int64), np.asarray(a, dtype=np.int64))
        self.r = np.zeros((self.K, self.S, self.A))
        self.q = np.zeros((self.K, self.S, self.A))
        for k in range(self.K):
            self.r[k], self.q[k] = iavi(self.T_env, expert[k], self.discount, self.iavi_tol, self.iavi_max_iters,
                                        self.eps)
        self.pis = softmax_policy(self.q)
        self._logpis_em = np.log(self.pis)
        return self

    def _emit(self, logpis, s, a, kin=None):
        """(N, T, K) emission LL (actions + kinematics); 0 on padded steps."""
        s, a = np.asarray(s, dtype=np.int64), np.asarray(a, dtype=np.int64)
        kll = self._kll(kin, s.shape)
        if kin is not None and (a < 0).any():
            kll = np.where((a >= 0)[..., None], kll, 0.0)
        return emission_loglik(logpis, s, a) + kll

    def _fb(self, s, a, w, c, logpis=None, kin=None):
        logpis = np.log(self.pis) if logpis is None else logpis
        lP = self.log_trans(self.logp_tr, w, c)
        return forward_backward_tv(self.logp_init, lP, self._emit(logpis, s, a, kin), step_mask(a))

    def loglik(self, s, a, w, c, kin=None):
        _, _, ll, _ = self._fb(s, a, w, c, kin=kin)
        return float(ll.sum() / int((np.asarray(a) >= 0).sum()))

    def step_loglik(self, s, a, w, c, kin=None):
        return self._fb(s, a, w, c, kin=kin)[3]

    def posteriors(self, s, a, w, c, kin=None):
        return np.exp(self._fb(s, a, w, c, kin=kin)[0])

    def decode(self, s, a, w, c, kin=None):
        s = np.asarray(s, dtype=np.int64)
        lP = self.log_trans(self.logp_tr, w, c)
        return viterbi_tv(self.logp_init, lP, self._emit(np.log(self.pis), s, a, kin), step_mask(a))

    def params(self):
        p = super().params()
        p["alpha"] = self.alpha
        p["coupling"] = self.coupling
        p["B"] = np.asarray(self.B)
        return p


def hiavi_step_loglik(m, s, a, kin=None):
    """One-step predictive LLs (N, T) of a plain HIAVI model (zero context)."""
    N, T = np.asarray(s).shape
    lP = np.broadcast_to(m.logp_tr, (N, max(T - 1, 0), m.K, m.K))
    return forward_backward_tv(m.logp_init, lP, m._lls(s, a, kin), step_mask(a))[3]


def fit_coupled(base_model, T_env, s, a, pc, tr_idx, settings, kern_tr, n_outer=5, alpha0=0.1, fit_alpha=True,
                verbose=False, kin=None, coupling="mixture", coupling_l2=1.0):
    """
    Mean-field coordinate ascent on the training sequences tr_idx (contexts from the other TRAINING animals).
    kern_tr = pc.build(tr_idx, tr_idx, ...). Returns the fitted CoupledHIAVI and the final training context.
    coupling / coupling_l2: see CoupledHIAVI (kern_tr["coupling"] from the run's variant overrides both).
    """
    cpl = dict(kern_tr.get("coupling") or {}) if isinstance(kern_tr, dict) else {}
    coupling, coupling_l2 = cpl.get("coupling", coupling), float(cpl.get("coupling_l2", coupling_l2))
    s_tr, a_tr = s[tr_idx], a[tr_idx]
    k_tr = None if kin is None else kin[tr_idx]
    settings = {k: v for k, v in settings.items() if k != "kinematics"}
    m = CoupledHIAVI(base_model.K, T_env, alpha0=alpha0, fit_alpha=fit_alpha, coupling=coupling,
                     coupling_l2=coupling_l2, **settings).init_from(base_model)
    gamma = base_model.posteriors(s_tr, a_tr, k_tr)
    for o in range(int(n_outer)):
        w, c = PartnerContext.context(kern_tr, gamma)
        m.fit_em(s_tr, a_tr, w, c, kin=k_tr)
        gamma = np.exp(m._fb(s_tr, a_tr, w, c, m._logpis_em, k_tr)[0])
        if verbose:
            print(f" outer {o}: train LL {m.ll_history[-1]:.5f} {m.coupling_stats()} "
                  f"(context on {100 * np.mean(w > 0):.1f}% of steps)")
    w, c = PartnerContext.context(kern_tr, gamma)
    m.finalize(s_tr, a_tr, w, c, kin=k_tr)
    return m, (w, c)


def _as_windows(window):
    """None -> None; (t0, t1) -> [(t0, t1)]; [(t0, t1), ...] -> list of float tuples."""
    if window is None:
        return None
    if len(window) and np.ndim(window[0]) == 0:
        return [(float(window[0]), float(window[1]))]
    return [(float(a), float(b)) for a, b in window]
