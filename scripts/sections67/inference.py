"""Statistical plan of protocol §5 and §9.5: stratified network bootstrap, H1-H4, Holm."""
import numpy as np
import pandas as pd
from scipy import stats


def holm(pvals):
    """Holm step-down adjusted p-values (monotone, capped at 1); NaN treated as 1."""
    p = np.array([1.0 if (v is None or np.isnan(v)) else float(v) for v in pvals])
    m = len(p)
    order = np.argsort(p, kind="stable")
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def null_centred_p(boot, obs):
    """(1 + #{|theta_b - theta_obs| >= |theta_obs|}) / (B + 1), over finite replicates.

    Returns NaN (not estimable) when the bootstrap distribution is degenerate (no spread).
    """
    b = np.asarray(boot, float)
    b = b[np.isfinite(b)]
    if not np.isfinite(obs) or len(b) == 0 or np.ptp(b) == 0:
        return np.nan
    return float((1 + np.sum(np.abs(b - obs) >= abs(obs))) / (len(b) + 1))


def min_per_subset(subsets):
    s = pd.Series(subsets)
    return int(s.value_counts().min()) if len(s) else 0


def subset_weights(subsets):
    """Equal total weight per subset, equal weight per network within subset."""
    s = pd.Series(subsets)
    n = s.map(s.value_counts())
    return (1.0 / n).to_numpy(float)


def weighted_subset_mean(values, subsets):
    v = np.asarray(values, float)
    s = np.asarray(subsets)
    levels = sorted(set(s))
    return float(np.mean([v[s == lv].mean() for lv in levels]))


def stratified_indices(subsets, n_boot, rng):
    """(n_boot, n) array of row indices resampled with replacement within each subset."""
    s = np.asarray(subsets)
    n = len(s)
    out = np.empty((n_boot, n), dtype=np.int64)
    for lv in sorted(set(s)):
        pos = np.where(s == lv)[0]
        out[:, pos] = pos[rng.integers(0, len(pos), size=(n_boot, len(pos)))]
    return out


def _design(C, subsets, levels, separate_slopes):
    s = np.asarray(subsets)
    dummies = np.column_stack([(s == lv).astype(float) for lv in levels])
    if separate_slopes:
        return np.column_stack([dummies, dummies * np.asarray(C, float)[:, None]])
    return np.column_stack([dummies, np.asarray(C, float)])


def wls(y, X, w):
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    return beta


def h1_slopes(dEN, dEBM, C, subsets, levels):
    """Common slope of Delta_m-fastai on C with one intercept per subset, for m = EN, EBM."""
    s = np.asarray(subsets)
    C = np.asarray(C, float)
    X = _design(C, s, levels, False)
    w = subset_weights(s)
    if np.ptp(C) == 0 or any((s == lv).sum() == 0 for lv in levels):
        return np.nan, np.nan
    within = sum(np.ptp(C[s == lv]) for lv in levels)
    if within == 0:
        return np.nan, np.nan
    return wls(np.asarray(dEN, float), X, w)[-1], wls(np.asarray(dEBM, float), X, w)[-1]


def h2_slopes(d, C, subsets, levels):
    s = np.asarray(subsets)
    C = np.asarray(C, float)
    out = {}
    for lv in levels:
        m = s == lv
        if m.sum() < 2 or np.ptp(C[m]) == 0:
            out[lv] = np.nan
        else:
            out[lv] = np.polyfit(C[m], np.asarray(d, float)[m], 1)[0]
    return out


def theilsen_slopes(d, C, subsets, levels):
    s = np.asarray(subsets)
    C = np.asarray(C, float)
    out = {}
    for lv in levels:
        m = s == lv
        if m.sum() < 2 or np.ptp(C[m]) == 0:
            out[lv] = np.nan
        else:
            out[lv] = stats.theilslopes(np.asarray(d, float)[m], C[m], method="separate")[0]
    return out


def cooks_distance(y, X, w):
    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    yw = y * sw
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    e = yw - Xw @ beta
    XtX_inv = np.linalg.pinv(Xw.T @ Xw)
    h = np.einsum("ij,jk,ik->i", Xw, XtX_inv, Xw)
    n, p = X.shape
    s2 = np.sum(e ** 2) / max(n - p, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        D = e ** 2 * h / (p * s2 * (1 - h) ** 2)
    return D


def percentile_ci(boot, q=(2.5, 97.5)):
    b = np.asarray(boot, float)
    b = b[np.isfinite(b)]
    if len(b) == 0:
        return np.nan, np.nan
    lo, hi = np.percentile(b, q)
    return float(lo), float(hi)


def describe_pair(d, subsets):
    d = np.asarray(d, float)
    return {"weighted_mean": weighted_subset_mean(d, subsets) if len(d) else np.nan,
            "median": float(np.median(d)) if len(d) else np.nan,
            "win_first": float(np.mean(d < 0)) if len(d) else np.nan,
            "win_second": float(np.mean(d > 0)) if len(d) else np.nan,
            "ties": float(np.mean(d == 0)) if len(d) else np.nan, "n": int(len(d))}
