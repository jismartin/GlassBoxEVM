"""Stage 'explain': local finite differences (§6.1), partial dependence and S1/ST (§6.2, §9.4).

The exact function g is evaluated once per network (``explain --model exact``, cached in
explain/exact/) and reused by every surrogate.  All samples, perturbations and permutations
are common to g and to all models; validity masks depend only on the reference and the domain.
"""
import os

import numpy as np
import pandas as pd

from . import common
from . import config as cfg
from . import data
from . import models
from .network import critical_keys, critical_matrix, interp_lin


# ------------------------------------------------------------------ helpers
def batched_eval(f, makers, max_rows=100000):
    """Evaluate f on blocks produced lazily by zero-argument callables (bounded memory).

    Blocks are concatenated up to max_rows rows per call; returns one result array per block.
    """
    out = []
    buf, sizes, rows = [], [], 0

    def flush():
        if not buf:
            return
        y = f(np.vstack(buf))
        pos = 0
        for n in sizes:
            out.append(y[pos:pos + n])
            pos += n
        del buf[:]
        del sizes[:]

    for mk in makers:
        B = mk()
        if buf and rows + B.shape[0] > max_rows:
            flush()
            rows = 0
        buf.append(B)
        sizes.append(B.shape[0])
        rows += B.shape[0]
    flush()
    return out


def hier_mean(values, valid):
    """Mean over valid scenarios per (activity, step), then over steps, then over activities."""
    import warnings
    v = np.where(valid, values, 0.0)
    n = valid.sum(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        per_step = np.where(n > 0, v.sum(axis=2) / np.maximum(n, 1), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        per_act = np.nanmean(per_step, axis=1) if per_step.size else np.zeros(0)
    per_act = np.asarray(per_act, float)
    ok = ~np.isnan(per_act)
    return (float(per_act[ok].mean()) if ok.any() else np.nan), per_step, per_act


# ------------------------------------------------------------------ design
class Design:
    """Samples and perturbations shared by g and all models for one network."""

    def __init__(self, ctx, network, need_sobol=True):
        b = data.network_bundle(ctx, network)
        self.b = b
        self.network = network
        E = cfg.EXPLAIN
        df, dcols = b["df"], b["dcols"]
        self.dcols = dcols
        self.D = df[dcols].to_numpy(float)
        self.T_all = b["net"].makespan(self.D)
        self.var_T = float(self.T_all.var())
        self.variable = np.where(~b["const"])[0]
        self.act_ids = np.array([int(c.replace("duration", "")) for c in dcols])
        Xtr = b["df_train"][dcols].to_numpy(float)
        self.sd = Xtr.std(axis=0, ddof=E["sd_ddof"])
        sup = b["support"]
        self.lo = sup["support_lo"].to_numpy(float)
        self.hi = sup["support_hi"].to_numpy(float)
        # sample positions refer to df rows (index == position for scenario files)
        assert (df.index.to_numpy() == np.arange(len(df))).all()
        self.local_idx = data.stage_indices(ctx.test_runs, network, "local_test", E["n_local"])
        self.pd_idx = data.stage_indices(ctx.test_runs, network, "pd_background", E["n_pd_background"])
        self.X0 = self.D[self.local_idx]
        steps = np.array(E["steps_sd"])
        self.steps = steps
        nv, ns, n = len(self.variable), len(steps), len(self.local_idx)
        self.delta = np.zeros((nv, ns, n))
        self.valid = np.zeros((nv, ns, n), bool)
        self.clipped = np.zeros((nv, ns, n), bool)
        for a, j in enumerate(self.variable):
            for s, h in enumerate(steps):
                target = self.X0[:, j] + h * self.sd[j]
                clip = np.clip(target, self.lo[j], self.hi[j])
                d = clip - self.X0[:, j]
                self.delta[a, s] = d
                self.clipped[a, s] = clip != target
                self.valid[a, s] = np.abs(d) >= E["min_effective_fraction"] * abs(h * self.sd[j])
        q = np.linspace(E["pd_q_lo"], E["pd_q_hi"], E["pd_n_grid"])
        self.pd_grid = [np.unique(np.quantile(self.D[:, j], q)) for j in self.variable]
        self.pd_bg = self.D[self.pd_idx]
        if need_sobol:
            sob, sc = data.load_extra_sample(ctx, network, "sobol_sample")
            assert list(sc) == list(dcols)
            self.S = sob[dcols].to_numpy(float)
            self.s1_idx = np.sort(common.stage_rng(network, "sobol_sample", 1).choice(
                len(self.S), size=min(E["n_s1_background"], len(self.S)), replace=False))
            self.s1_grid = [np.unique(np.quantile(self.S[:, j], q)) for j in self.variable]
            self.perms = []
            for r in range(E["n_perm_seeds"]):
                rng = common.stage_rng(network, "sobol_perm", r)
                self.perms.append([rng.permutation(len(self.S)) for _ in self.variable])

    def perturbed_makers(self):
        makers = []
        for a, j in enumerate(self.variable):
            for s in range(len(self.steps)):
                makers.append(self._pert_maker(a, j, s))
        return makers

    def _pert_maker(self, a, j, s):
        def mk():
            Xp = self.X0.copy()
            Xp[:, j] = self.X0[:, j] + self.delta[a, s]
            return Xp
        return mk


def _set_col_maker(B0, j, v):
    def mk():
        B = B0.copy()
        B[:, j] = v
        return B
    return mk


def _perm_maker(S, j, perm):
    def mk():
        Sp = S.copy()
        Sp[:, j] = S[perm, j]
        return Sp
    return mk


def function_outputs(f, dz, with_sobol=True):
    """All quantities of §6.1-6.2 for one function f (g or a surrogate)."""
    nv, ns = len(dz.variable), len(dz.steps)
    out = {}
    f0 = f(dz.X0)
    pert = batched_eval(f, dz.perturbed_makers())
    out["f0"] = f0
    out["dy"] = np.stack([p - f0 for p in pert]).reshape(nv, ns, -1)
    makers = [_set_col_maker(dz.pd_bg, j, v) for a, j in enumerate(dz.variable) for v in dz.pd_grid[a]]
    vals = batched_eval(f, makers)
    k = 0
    out["pd"] = []
    for a in range(nv):
        m = len(dz.pd_grid[a])
        out["pd"].append(np.array([vals[k + i].mean() for i in range(m)]))
        k += m
    if with_sobol:
        fS = f(dz.S)
        varf = float(fS.var())
        out["varf_sobol"] = varf
        Bs = dz.S[dz.s1_idx]
        makers = [_set_col_maker(Bs, j, v) for a, j in enumerate(dz.variable) for v in dz.s1_grid[a]]
        vals = batched_eval(f, makers)
        k = 0
        s1 = np.full(nv, np.nan)
        for a, j in enumerate(dz.variable):
            m = len(dz.s1_grid[a])
            pdv = np.array([vals[k + i].mean() for i in range(m)])
            k += m
            if varf > 0:
                s1[a] = interp_lin(dz.S[:, j], dz.s1_grid[a], pdv).var() / varf
        st = np.full((nv, len(dz.perms)), np.nan)
        for r, perm in enumerate(dz.perms):
            makers = [_perm_maker(dz.S, j, perm[a]) for a, j in enumerate(dz.variable)]
            ys = batched_eval(f, makers)
            if varf > 0:
                st[:, r] = [np.mean((fS - yp) ** 2) / (2 * varf) for yp in ys]
        out["S1"] = s1
        out["ST"] = st
    return out


def centred_curve(x, grid, pdv):
    c = interp_lin(x, grid, pdv)
    return c - c.mean()


def curvature(x, c, var_T):
    X = np.c_[np.ones(len(x)), x]
    b = np.linalg.lstsq(X, c, rcond=None)[0]
    return float(np.mean((c - X @ b) ** 2) / var_T) if var_T > 0 else np.nan


# ------------------------------------------------------------------ exact reference
def exact_outputs(ctx, network, dz):
    out = ctx.stage_dir(network, "explain", "exact")
    inputs = {"scenarios_sha256": data.scenario_input_hash(ctx, network),
              "sobol_sha256": _sobol_hash(ctx, network)}
    st = common.Stamp(out, inputs)
    npz = os.path.join(out, "exact_reference.npz")
    if not st.begin(ctx.force):
        z = np.load(npz, allow_pickle=False)
        res = {k: z[k] for k in ("f0", "dy", "S1", "ST", "switch", "T0")}
        res["varf_sobol"] = float(z["varf_sobol"])
        res["pd"] = [z["pd_%d" % a] for a in range(len(dz.variable))]
        res["_computed"] = False
        return res
    net = dz.b["net"]
    res = function_outputs(net.makespan, dz)
    # critical-set switches under each perturbation (exact function only)
    tol = cfg.STRUCTURE["critical_tol"]
    TF0, T0 = net.floats(dz.X0)
    k0 = critical_keys(critical_matrix(TF0, T0, tol))
    sw = np.zeros(res["dy"].shape, bool)
    for i, mk in enumerate(dz.perturbed_makers()):
        Xp = mk()
        TF, T = net.floats(Xp)
        k = critical_keys(critical_matrix(TF, T, tol))
        sw.reshape(-1, Xp.shape[0])[i] = k != k0
    res["switch"] = sw
    res["T0"] = T0
    arrays = {k: res[k] for k in ("f0", "dy", "S1", "ST", "switch", "T0")}
    arrays["varf_sobol"] = np.array(res["varf_sobol"])
    arrays["local_idx"] = dz.local_idx
    arrays["pd_idx"] = dz.pd_idx
    arrays["s1_idx"] = dz.s1_idx
    for a in range(len(dz.variable)):
        arrays["pd_%d" % a] = res["pd"][a]
    common.atomic_save_npz(npz, **arrays)
    common.atomic_write_csv(pd.DataFrame({"stage": "local_test", "scenario_id": dz.local_idx}),
                            os.path.join(out, "indices_local_test.csv"))
    common.atomic_write_csv(pd.DataFrame({"stage": "pd_background", "scenario_id": dz.pd_idx}),
                            os.path.join(out, "indices_pd_background.csv"))
    common.atomic_write_csv(pd.DataFrame({"stage": "sobol_sample_s1_background", "row": dz.s1_idx}),
                            os.path.join(out, "indices_s1_background.csv"))
    _write_model_tables(ctx, network, "exact", dz, res, res, out)
    st.finish(info={"n_variable": int(len(dz.variable))})
    res["_computed"] = True
    return res


def _sobol_hash(ctx, network):
    base = data.simulated_base(ctx, network, "sobol_sample")
    for e in (".parquet", ".csv.gz"):
        if os.path.exists(base + e):
            return common.sha256_file(base + e)
    raise FileNotFoundError(base + " (run the simulate stage first)")


# ------------------------------------------------------------------ metrics
def local_metrics(dz, mo, ex):
    E = cfg.EXPLAIN
    valid = dz.valid
    with np.errstate(divide="ignore", invalid="ignore"):
        Dm = mo["dy"] / dz.delta
        Dg = ex["dy"] / dz.delta
    err = np.abs(Dm - Dg)
    terr = np.abs(mo["dy"] - ex["dy"])
    tol = E["violation_tol"]
    viol = ((Dm < -tol) | (Dm > 1 + tol)).astype(float)
    mae_d, per_step, per_act = hier_mean(err, valid)
    mae_t, _, _ = hier_mean(terr, valid)
    vr, _, _ = hier_mean(viol, valid)
    null = np.abs(ex["dy"]) < E["null_change_rel_tol"] * ex["T0"][None, None, :]
    vn = valid & null
    vnn = valid & ~null
    row = {"MAE_D": mae_d, "MAE_time": mae_t, "violation_rate": vr,
           "violation_rate_pooled": float(viol[valid].mean()) if valid.any() else np.nan,
           "valid_n": int(valid.sum()), "clipped_n": int(dz.clipped.sum()),
           "excluded_n": int((~valid).sum()), "switch_n": int((ex["switch"] & valid).sum()),
           "null_n": int(vn.sum()), "nonnull_n": int(vnn.sum()),
           "MAE_D_null_pooled": float(err[vn].mean()) if vn.any() else np.nan,
           "MAE_D_nonnull_pooled": float(err[vnn].mean()) if vnn.any() else np.nan,
           "MAE_D_switch_pooled": float(err[valid & ex["switch"]].mean()) if (valid & ex["switch"]).any() else np.nan,
           "n_activities": int(len(dz.variable)),
           "n_activities_no_valid": int(np.sum(valid.sum(axis=(1, 2)) == 0)),
           "n_local": int(len(dz.local_idx)),
           "exact_D_min": float(np.nanmin(np.where(valid, Dg, np.nan))) if valid.any() else np.nan,
           "exact_D_max": float(np.nanmax(np.where(valid, Dg, np.nan))) if valid.any() else np.nan}
    rows = []
    for a, j in enumerate(dz.variable):
        for s, h in enumerate(dz.steps):
            v = valid[a, s]
            rows.append({"activity_id": int(dz.act_ids[j]), "step_sd": float(h), "valid_n": int(v.sum()),
                         "clipped_n": int(dz.clipped[a, s].sum()),
                         "mean_abs_D_err": float(err[a, s][v].mean()) if v.any() else np.nan,
                         "mean_abs_time_err": float(terr[a, s][v].mean()) if v.any() else np.nan,
                         "mean_D_model": float(Dm[a, s][v].mean()) if v.any() else np.nan,
                         "mean_D_exact": float(Dg[a, s][v].mean()) if v.any() else np.nan,
                         "violation_n": int(viol[a, s][v].sum()),
                         "switch_n": int((ex["switch"][a, s] & v).sum()),
                         "null_n": int((null[a, s] & v).sum())})
    return row, pd.DataFrame(rows)


def _write_model_tables(ctx, network, model_id, dz, mo, ex, out):
    lrow, ldet = local_metrics(dz, mo, ex)
    srows, crows = [], []
    pd_err = []
    for a, j in enumerate(dz.variable):
        x = dz.D[:, j]
        cm = centred_curve(x, dz.pd_grid[a], mo["pd"][a])
        cg = centred_curve(x, dz.pd_grid[a], ex["pd"][a])
        e = float(np.mean((cm - cg) ** 2) / dz.var_T) if dz.var_T > 0 else np.nan
        pd_err.append(e)
        var_g_curve = float(cg.var())
        r2 = float(1 - np.mean((cm - cg) ** 2) / var_g_curve) if var_g_curve > 0 else np.nan
        for r in range(mo["ST"].shape[1]):
            srows.append({"cohort": ctx.cohort, "network_id": network, "model_id": model_id,
                          "activity_id": int(dz.act_ids[j]), "PD_error": e, "PD_R2": r2,
                          "exact_pd_constant": bool(var_g_curve == 0),
                          "PD_curvature_model": curvature(x, cm, dz.var_T),
                          "PD_curvature_exact": curvature(x, cg, dz.var_T),
                          "S1": float(mo["S1"][a]), "ST": float(mo["ST"][a, r]),
                          "permutation_seed": int(common.derive_seed(network, "sobol_perm", r)),
                          "permutation_replicate": r, "var_function_sobol": mo["varf_sobol"],
                          "function_source": model_id})
        for gi, gv in enumerate(dz.pd_grid[a]):
            crows.append({"activity_id": int(dz.act_ids[j]), "grid_point": gi, "duration": gv,
                          "pd_model": mo["pd"][a][gi], "pd_exact": ex["pd"][a][gi]})
    sens = pd.DataFrame(srows)
    pe = np.array(pd_err)
    lrow.update(cohort=ctx.cohort, network_id=network, model_id=model_id,
                PD_error_mean=float(np.nanmean(pe)) if len(pe) and not np.isnan(pe).all() else np.nan,
                var_T_emp=dz.var_T, var_function_sobol=mo["varf_sobol"])
    front = ["cohort", "network_id", "model_id"]
    lm = pd.DataFrame([lrow])
    lm = lm[front + [c for c in lm.columns if c not in front]]
    common.atomic_write_csv(lm, os.path.join(out, "local_metrics.csv"))
    ldet.insert(0, "model_id", model_id)
    ldet.insert(0, "network_id", network)
    common.write_table(ldet, os.path.join(out, "local_detail"))
    common.write_table(sens, os.path.join(out, "sensitivity"))
    common.write_table(pd.DataFrame(crows).assign(network_id=network, model_id=model_id),
                       os.path.join(out, "pd_curves"))
    return lrow


def run(ctx, network, model_id):
    dz = Design(ctx, network)
    ex = exact_outputs(ctx, network, dz)
    if model_id == "exact":
        return {"status": "ok"} if ex["_computed"] else None
    fit_dir = ctx.stage_dir(network, "fit", model_id)
    stamp_fit = os.path.join(fit_dir, "_stamp.json")
    if not os.path.exists(stamp_fit):
        raise RuntimeError("fit stage missing for %s %s" % (network, model_id))
    out = ctx.stage_dir(network, "explain", model_id)
    inputs = {"fit_stamp_sha256": common.sha256_file(stamp_fit),
              "exact_stamp_sha256": common.sha256_file(os.path.join(ctx.stage_dir(network, "explain", "exact"),
                                                                    "_stamp.json"))}
    st = common.Stamp(out, inputs)
    if not st.begin(ctx.force):
        return None
    pipe, meta = models.load(fit_dir, model_id)
    assert meta["column_order"] == dz.dcols
    mo = function_outputs(pipe.predict, dz)
    lrow = _write_model_tables(ctx, network, model_id, dz, mo, ex, out)
    return st.finish(info={"MAE_D": lrow["MAE_D"]})
