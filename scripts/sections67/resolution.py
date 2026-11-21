"""Stage 'resolution': numerical resolution checks run in the pilot (protocol §7.1).

For one network, compares the default settings with finer or alternative ones:

* empirical decomposition L_emp / I_int_emp / Q_emp with 21- and 41-point grids;
* the same decomposition evaluated on an independent simulated sample (the stage
  'sobol_sample', independent of the training and test scenarios), and Q_emp compared with
  a direct estimate of single-activity curvature;
* first-order Sobol indices of g with 21- and 41-point grids and 500- and 2000-row
  backgrounds;
* partial-dependence error of the focal models with the default grid, a 41-point grid and
  a grid that adds the support limits of each activity.

Results are written to resolution/resolution.csv in long format
(check, function, setting, quantity, value).
"""
import os

import numpy as np
import pandas as pd

from . import common
from . import config as cfg
from . import data
from . import models
from .explain import Design, _set_col_maker, batched_eval, centred_curve, curvature
from .network import interp_lin
from .structure import exact_pd


def _grid(x, n, lo=None, hi=None):
    E = cfg.EXPLAIN
    g = np.quantile(x, np.linspace(E["pd_q_lo"], E["pd_q_hi"], n))
    if lo is not None:
        g = np.concatenate([[lo], g, [hi]])
    return np.unique(g)


def _pd(f, B, j, grid):
    makers = [_set_col_maker(B, j, v) for v in grid]
    return np.array([y.mean() for y in batched_eval(f, makers)])


def _decomposition(net, D, T, variable, n_grid):
    """Same estimator as structure.decomposition, with an n_grid-point quantile grid."""
    varT = T.var()
    X = np.c_[np.ones(len(T)), D[:, variable]]
    b = np.linalg.lstsq(X, T, rcond=None)[0]
    with np.errstate(all="ignore"):
        L = float(np.mean((T - X @ b) ** 2) / varT)
    gadd = np.full(len(T), T.mean())
    curves = {}
    q_direct = 0.0
    for j in variable:
        grid = _grid(D[:, j], n_grid)
        pdv = exact_pd(net, D, j, grid, D)
        f = interp_lin(D[:, j], grid, pdv)
        curves[j] = (grid, pdv, float(f.mean()))
        gadd += f - f.mean()
        q_direct += curvature(D[:, j], f - f.mean(), varT)
    I = float(np.mean((T - gadd) ** 2) / varT)
    return {"L": L, "I_int": I, "Q": L - I, "Q_direct": float(q_direct)}, b, curves


def run(ctx, network):
    out = ctx.stage_dir(network, "resolution")
    R = cfg.RESOLUTION
    fit_stamps = {}
    for m in R["models"]:
        p = os.path.join(ctx.stage_dir(network, "fit", m), "_stamp.json")
        if not os.path.exists(p):
            raise RuntimeError("fit stage missing for %s %s" % (network, m))
        fit_stamps[m] = common.sha256_file(p)
    dz = Design(ctx, network)
    inputs = {"scenarios_sha256": data.scenario_input_hash(ctx, network), "fit_stamps": fit_stamps}
    st = common.Stamp(out, inputs)
    if not st.begin(ctx.force):
        return None
    net = dz.b["net"]
    D, T, variable = dz.D, dz.T_all, dz.variable
    if len(variable) == 0 or T.var() == 0:
        raise RuntimeError("no variable activities or Var(g) = 0 for %s" % network)
    lo, hi = dz.lo, dz.hi
    rows = []

    def add(check, function, setting, quantity, value):
        rows.append({"cohort": ctx.cohort, "network_id": network, "check": check, "function": function,
                     "setting": setting, "quantity": quantity, "value": float(value)})

    # 1. decomposition: grid 21 vs 41 (in-sample, all scenarios)
    base = None
    for n in R["grids"]:
        dec, b, curves = _decomposition(net, D, T, variable, n)
        for k, v in dec.items():
            add("decomposition", "exact", "grid%d" % n, k, v)
        if base is None:
            base = (b, curves)
    # 2. decomposition evaluated on the independent sample with the default-grid estimates
    b, curves = base
    S = dz.S
    TS = net.makespan(S)
    varS = TS.var()
    XS = np.c_[np.ones(len(TS)), S[:, variable]]
    gaddS = np.full(len(TS), T.mean())
    for j in variable:
        grid, pdv, fmean = curves[j]
        gaddS += interp_lin(S[:, j], grid, pdv) - fmean
    with np.errstate(all="ignore"):
        L_ind = float(np.mean((TS - XS @ b) ** 2) / varS)
    I_ind = float(np.mean((TS - gaddS) ** 2) / varS)
    add("decomposition", "exact", "independent_sample", "L", L_ind)
    add("decomposition", "exact", "independent_sample", "I_int", I_ind)
    add("decomposition", "exact", "independent_sample", "Q", L_ind - I_ind)
    # 3. first-order Sobol indices of g: grid and background size
    rng = common.stage_rng(network, "sobol_sample", 2)
    bg_big = np.sort(rng.choice(len(S), size=min(R["s1_backgrounds"][1], len(S)), replace=False))
    settings = {"grid%d_bg%d" % (R["grids"][0], R["s1_backgrounds"][0]): (R["grids"][0], dz.s1_idx),
                "grid%d_bg%d" % (R["grids"][1], R["s1_backgrounds"][0]): (R["grids"][1], dz.s1_idx),
                "grid%d_bg%d" % (R["grids"][0], R["s1_backgrounds"][1]): (R["grids"][0], bg_big)}
    for name, (n, idx) in settings.items():
        B = S[idx]
        s1 = []
        for j in variable:
            grid = _grid(S[:, j], n)
            s1.append(interp_lin(S[:, j], grid, _pd(net.makespan, B, j, grid)).var() / varS)
        s1 = np.array(s1)
        add("sobol_S1", "exact", name, "sum_S1", s1.sum())
        for a, j in enumerate(variable):
            add("sobol_S1", "exact", name, "S1_activity_%d" % dz.act_ids[j], s1[a])
    # 4. partial-dependence error of the focal models: grid and support limits
    funcs = {"exact": net.makespan}
    for m in R["models"]:
        pipe, meta = models.load(ctx.stage_dir(network, "fit", m), m)
        assert meta["column_order"] == dz.dcols
        funcs[m] = pipe.predict
    grids = {"grid%d" % R["grids"][0]: [_grid(D[:, j], R["grids"][0]) for j in variable],
             "grid%d" % R["grids"][1]: [_grid(D[:, j], R["grids"][1]) for j in variable]}
    if R["support_limits_grid"]:
        grids["grid%d_support" % R["grids"][0]] = [_grid(D[:, j], R["grids"][0], lo[j], hi[j]) for j in variable]
    for gname, gl in grids.items():
        exact_c = []
        for a, j in enumerate(variable):
            exact_c.append(centred_curve(D[:, j], gl[a], _pd(funcs["exact"], dz.pd_bg, j, gl[a])))
        for m in R["models"]:
            errs = []
            for a, j in enumerate(variable):
                cm = centred_curve(D[:, j], gl[a], _pd(funcs[m], dz.pd_bg, j, gl[a]))
                errs.append(np.mean((cm - exact_c[a]) ** 2) / dz.var_T)
            add("pd_error", m, gname, "PD_error_mean", np.mean(errs))
    common.atomic_write_csv(pd.DataFrame(rows), os.path.join(out, "resolution.csv"))
    return st.finish(info={"n_rows": len(rows)})
