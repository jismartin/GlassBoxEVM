"""Exact project-duration function g, vectorised CPM and critical-set measures.

``Net`` is ported from the exploratory ``ref_sens.py``; ``critical_set_stats`` from
``struct_measures.py`` (C = 1 - frequency of the modal critical set, tolerance relative to the
scenario makespan, ties included because the set holds every activity with float < tol*T).
"""
import networkx as nx
import numpy as np


def interp_lin(x, g, v):
    """Piecewise-linear interpolation with linear (not flat) extrapolation beyond the grid ends."""
    x = np.asarray(x, float)
    y = np.interp(x, g, v)
    if len(g) < 2:
        return y
    lo = x < g[0]
    hi = x > g[-1]
    if g[1] > g[0]:
        y[lo] = v[0] + (x[lo] - g[0]) * (v[1] - v[0]) / (g[1] - g[0])
    if g[-1] > g[-2]:
        y[hi] = v[-1] + (x[hi] - g[-1]) * (v[-1] - v[-2]) / (g[-1] - g[-2])
    return y


class Net:
    """Forward/backward CPM over a matrix of durations (rows = scenarios, columns = dcols)."""

    def __init__(self, G, dcols):
        self.dcols = list(dcols)
        self.col = {n: self.dcols.index("duration%s" % n) for n in G.nodes()}
        self.order = list(nx.topological_sort(G))
        self.pred = {n: [self.col[q] for q in G.predecessors(n)] for n in G.nodes()}
        self.succ = {n: [self.col[q] for q in G.successors(n)] for n in G.nodes()}

    def forward(self, D):
        D = np.asarray(D, float)
        ES = np.zeros_like(D)
        EF = np.zeros_like(D)
        for n in self.order:
            c = self.col[n]
            p = self.pred[n]
            if p:
                ES[:, c] = EF[:, p].max(1)
            EF[:, c] = ES[:, c] + D[:, c]
        return ES, EF, EF.max(1)

    def makespan(self, D, chunk=200000):
        D = np.asarray(D, float)
        if D.shape[0] <= chunk:
            return self.forward(D)[2]
        return np.concatenate([self.forward(D[i:i + chunk])[2] for i in range(0, D.shape[0], chunk)])

    def floats(self, D):
        D = np.asarray(D, float)
        ES, EF, T = self.forward(D)
        LF = np.repeat(T[:, None], D.shape[1], 1)
        for n in reversed(self.order):
            s = self.succ[n]
            if s:
                LF[:, self.col[n]] = (LF[:, s] - D[:, s]).min(1)
        return LF - EF, T


def critical_matrix(TF, T, tol):
    return TF < tol * T[:, None]


def critical_set_stats(crit):
    """C, number of distinct sets, Shannon entropy H, N_eff=exp(H), modal frequency."""
    keys = np.packbits(crit, axis=1)
    _, inv, cnt = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    ps = cnt / cnt.sum()
    H = float(-(ps * np.log(ps)).sum())
    return {"C": float(1 - ps.max()), "n_sets": int(len(cnt)), "H": H, "N_eff": float(np.exp(H)),
            "modal_freq": float(ps.max())}, np.asarray(inv).ravel(), cnt


def critical_keys(crit):
    """Row-wise hashable keys of critical sets (bytes of packed bits)."""
    keys = np.packbits(crit, axis=1)
    return np.array([k.tobytes() for k in keys], dtype=object)
