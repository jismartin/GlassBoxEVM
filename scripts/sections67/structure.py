"""Stage 'structure': exact function check, C, CI, N_eff, topology, OLS RelMSE, Q/I_int (§4)."""
import os

import numpy as np
import pandas as pd

from . import common
from . import config as cfg
from . import data
from .network import critical_matrix, critical_set_stats, interp_lin

TOPO_NAMES = ["n_act", "CNC", "OS", "SP", "AD", "LA", "I5", "TF"]


def ols_relmse(D, T, test_runs, variable):
    """Out-of-sample OLS RelMSE on the fixed split (ref_sens.py): SSE_OLS / SSE_mean(y_train)."""
    mask = np.zeros(len(T), bool)
    mask[test_runs] = True
    X = D[:, variable]
    b = np.linalg.lstsq(np.c_[np.ones((~mask).sum()), X[~mask]], T[~mask], rcond=None)[0]
    pred = np.c_[np.ones(mask.sum()), X[mask]] @ b
    den = np.sum((T[~mask].mean() - T[mask]) ** 2)
    return float(np.sum((pred - T[mask]) ** 2) / den) if den > 0 else np.nan


def exact_pd(net, D, j, grid, background):
    vals = np.empty(len(grid))
    for k, v in enumerate(grid):
        B = background.copy()
        B[:, j] = v
        vals[k] = net.makespan(B).mean()
    return vals


def decomposition(net, D, T, variable):
    """Empirical L_emp, I_int_emp, Q_emp (§4.2), ported from pair_decomp.py/struct_measures.py."""
    dc = cfg.STRUCTURE["decomposition"]
    varT = T.var()
    if varT == 0:
        return {"L_emp": np.nan, "I_int_emp": np.nan, "Q_emp": np.nan, "Q_over_L": np.nan,
                "validity_flag": "var_g_zero"}
    X = np.c_[np.ones(len(T)), D[:, variable]]
    b = np.linalg.lstsq(X, T, rcond=None)[0]
    with np.errstate(all="ignore"):  # spurious Accelerate matmul warnings on macOS
        L = float(np.mean((T - X @ b) ** 2) / varT)
    q = np.linspace(dc["q_lo"], dc["q_hi"], dc["n_grid"])
    gadd = np.full(len(T), T.mean())
    for j in variable:
        grid = np.unique(np.quantile(D[:, j], q))
        pdv = exact_pd(net, D, j, grid, D)
        f = interp_lin(D[:, j], grid, pdv)
        gadd += f - f.mean()
    I = float(np.mean((T - gadd) ** 2) / varT)
    Q = L - I
    ok = L >= dc["min_L_for_ratio"]
    return {"L_emp": L, "I_int_emp": I, "Q_emp": Q, "Q_over_L": Q / L if ok else np.nan,
            "validity_flag": "ok" if ok else "L_below_min_for_ratio"}


def run(ctx, network, with_decomposition=None):
    out = ctx.stage_dir(network, "structure")
    if with_decomposition is None:
        with_decomposition = cfg.STRUCTURE["decomposition"]["enabled"]
    inputs = {"scenarios_sha256": data.scenario_input_hash(ctx, network),
              "sm_sha256": common.sha256_file(ctx.sm_path(network)),
              "decomposition": bool(with_decomposition)}
    st = common.Stamp(out, inputs)
    if not st.begin(ctx.force):
        return None
    b = data.network_bundle(ctx, network)
    df, dcols, net, G = b["df"], b["dcols"], b["net"], b["G"]
    D = df[dcols].to_numpy(float)
    y = df["actual_duration"].to_numpy(float)
    TF, T = net.floats(D)
    g_check = float(np.max(np.abs(T - y)))
    if g_check > 1e-8:
        raise RuntimeError("exact function does not reproduce actual_duration (%g)" % g_check)
    variable = np.where(np.ptp(D, axis=0) > 0)[0]
    stats_main, inv, cnt = critical_set_stats(critical_matrix(TF, T, cfg.STRUCTURE["critical_tol"]))
    crit = critical_matrix(TF, T, cfg.STRUCTURE["critical_tol"])
    CI = crit.mean(0)
    lo, hi = cfg.STRUCTURE["intermediate_ci"]
    ci_var = CI[variable]
    topo = ctx.tools.compute_topological_indicators(G)
    row = {"cohort": ctx.cohort, "subset": b["subset"], "network_id": network,
           "C": stats_main["C"], "n_sets": stats_main["n_sets"], "H": stats_main["H"],
           "N_eff": stats_main["N_eff"], "modal_freq": stats_main["modal_freq"],
           "CI_share_intermediate": float(np.mean((ci_var > lo) & (ci_var < hi))) if len(ci_var) else np.nan,
           "CI_n_intermediate": int(np.sum((ci_var > lo) & (ci_var < hi))),
           "CI_mean_variable": float(ci_var.mean()) if len(ci_var) else np.nan,
           "CI_n_always_critical_variable": int(np.sum(ci_var == 1)),
           "OLS_RelMSE": ols_relmse(D, T, ctx.test_runs, variable),
           "n_variable": int(len(variable)), "n_activities": len(dcols) - 2,
           "mean_T": float(T.mean()), "var_T": float(T.var()), "g_check_max_abs": g_check}
    for tol in cfg.STRUCTURE["sensitivity_tols"]:
        s, _, _ = critical_set_stats(critical_matrix(TF, T, tol))
        row["C_tol%g" % tol] = s["C"]
    for name, v in zip(TOPO_NAMES, topo):
        row[name if name in ("SP", "TF") else "topo_" + name] = float(v)
    act = pd.DataFrame({"cohort": ctx.cohort, "network_id": network,
                        "activity_id": [int(c.replace("duration", "")) for c in dcols],
                        "variable": np.ptp(D, axis=0) > 0, "status": b["support"]["status"].to_numpy(),
                        "CI": CI, "mean": D.mean(0), "sd": D.std(0)})
    sets = pd.DataFrame({"set_rank": np.arange(len(cnt)), "count": cnt})
    sets["activities"] = [";".join(str(a) for a in act["activity_id"][crit[np.argmax(inv == k)]])
                          for k in range(len(cnt))]
    sets = sets.sort_values("count", ascending=False)
    common.atomic_write_csv(act, os.path.join(out, "activity_structure.csv"))
    common.atomic_write_csv(sets, os.path.join(out, "critical_sets.csv"))
    if with_decomposition:
        dec = decomposition(net, D, T, variable)
        dec.update(network_id=network, cohort=ctx.cohort, resolution="not_validated",
                   grid=cfg.STRUCTURE["decomposition"]["n_grid"], background_n=len(T))
        common.atomic_write_csv(pd.DataFrame([dec]), os.path.join(out, "decomposition.csv"))
    common.atomic_write_csv(pd.DataFrame([row]), os.path.join(out, "structure_metrics.csv"))
    return st.finish(info={"C": row["C"]})
