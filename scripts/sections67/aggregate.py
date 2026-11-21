"""Stage 'aggregate': collect per-network outputs, build network metrics, run the inference."""
import glob
import json
import os

import numpy as np
import pandas as pd

from . import common
from . import config as cfg
from . import inference as inf

# Operational choices that differ from, or make concrete, the protocol text (§9).
DEVIATIONS = [
    "Configuration exported to scripts/sections67/config.py (Python literals + SHA-256 config hash) "
    "instead of configs/sections_6_7.yaml: YAML/JSON cannot be placed in the repository with the "
    "available tooling.",
    "run_manifest is written as run_manifest.json (on disk, results directory) and run_manifest.md "
    "(repository-friendly copy).",
    "Detail tables (predictions, sensitivity, local_detail, pd_curves, simulated scenarios) are Parquet "
    "when pyarrow is installed and gzip CSV otherwise (pyarrow is not in the reference environment).",
    "S1: partial dependence evaluated on a 500-row background drawn from sobol_sample (stage "
    "'sobol_sample', replicate 1), grid of 21 quantiles of the sobol sample, variance over all "
    "sobol_sample rows; background size to be fixed in the pilot.",
    "Local violation rate uses the same three-level averaging as MAE_D; a pooled rate is also stored.",
]


def _current(paths):
    """Drop anything inside a '<dir>.superseded-<time>' folder."""
    return sorted(p for p in paths if ".superseded-" not in p)


def _collect(pattern):
    files = _current(glob.glob(pattern))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def _collect_tables(cohort_dir, stage, name):
    frames = []
    for d in _current(glob.glob(os.path.join(cohort_dir, "networks", "*", stage, "*"))):
        base = os.path.join(d, name)
        if common.table_exists(base):
            frames.append(common.read_table(base))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def collect(ctx):
    cd = ctx.cohort_dir
    res = {
        "fit_manifest": _collect(os.path.join(cd, "networks", "*", "fit", "*", "fit_manifest.csv")),
        "structure_metrics": _collect(os.path.join(cd, "networks", "*", "structure", "structure_metrics.csv")),
        "decomposition": _collect(os.path.join(cd, "networks", "*", "structure", "decomposition.csv")),
        "local_metrics": _collect(os.path.join(cd, "networks", "*", "explain", "*", "local_metrics.csv")),
        "decisions": _collect(os.path.join(cd, "networks", "*", "decide", "*", "decisions.csv")),
        "sensitivity": _collect_tables(cd, "explain", "sensitivity"),
    }
    return res


def network_metrics(tabs):
    sm = tabs["structure_metrics"]
    if sm.empty:
        raise RuntimeError("no structure_metrics found")
    nm = sm[["cohort", "subset", "network_id", "C", "OLS_RelMSE", "SP", "TF", "n_variable", "N_eff",
             "CI_share_intermediate"]].copy()
    fm = tabs["fit_manifest"]
    if not fm.empty:
        ok = fm[fm["fit_status"] == "ok"]
        w = ok.pivot_table(index="network_id", columns="model_id", values="RelMSE", aggfunc="first")
        w.columns = ["RelMSE_" + c for c in w.columns]
        nm = nm.merge(w, left_on="network_id", right_index=True, how="left")
    lm = tabs["local_metrics"]
    if not lm.empty:
        w = lm[lm["model_id"] != "exact"].pivot_table(index="network_id", columns="model_id",
                                                     values=["MAE_D", "violation_rate", "PD_error_mean"],
                                                     aggfunc="first")
        w.columns = ["%s_%s" % (a, b) for a, b in w.columns]
        nm = nm.merge(w, left_on="network_id", right_index=True, how="left")
    dec = tabs["decisions"]
    if not dec.empty:
        k = cfg.DECISION["h4_k"]
        d3 = dec[dec["k"] == k]
        ex = d3[d3["policy"] == "exact_greedy"].set_index("network_id")
        elig = dec.groupby("network_id")["n_eligible"].max()
        nm["n_eligible"] = nm["network_id"].map(elig)
        nm["baseline_mean_eval"] = nm["network_id"].map(ex["baseline_mean_eval"]) if len(ex) else np.nan
        for m in cfg.ALL_MODELS:
            pm = d3[d3["policy"] == m + "_greedy"].set_index("network_id")
            if len(pm) and len(ex):
                nm["regret%d_%s" % (k, m)] = nm["network_id"].map(ex["B_mean"] - pm["B_mean"])
    sens = tabs["sensitivity"]
    if not sens.empty:
        s1 = sens[sens["permutation_replicate"] == 0]
        ex = s1[s1["model_id"] == "exact"].set_index(["network_id", "activity_id"])["S1"]
        for m in cfg.FOCAL_MODELS:
            sm_ = s1[s1["model_id"] == m].set_index(["network_id", "activity_id"])["S1"]
            if len(sm_) == 0:
                continue
            j = pd.concat([sm_.rename("m"), ex.rename("g")], axis=1, join="inner").reset_index()
            rho = {}
            for nid, g in j.groupby("network_id"):
                if g["m"].nunique() > 1 and g["g"].nunique() > 1:
                    rho[nid] = g["m"].rank().corr(g["g"].rank())
                else:
                    rho[nid] = np.nan       # constant indices: correlation undefined, reported apart
            nm["S1_spearman_" + m] = nm["network_id"].map(rho)
    return nm


def _row(cohort, hyp, statistic, est, lo, hi, p, sub_counts, **extra):
    r = {"cohort": cohort, "hypothesis": hyp, "statistic": statistic, "estimate": est, "ci_low": lo,
         "ci_high": hi, "p_raw": p, "p_holm": np.nan,
         "n_by_subset": ";".join("%s:%d" % (k, v) for k, v in sorted(sub_counts.items()))}
    r.update(extra)
    return r


def run_inference(cohort, nm, n_boot=None, seed_network_id=None):
    n_boot = n_boot or cfg.INFERENCE["n_boot"]
    rng = common.stage_rng(seed_network_id or ("cohort:" + cohort), "network_bootstrap", 0)
    levels_all = [s for s in cfg.COHORTS["subsets"] if s in set(nm["subset"])]
    rows = []
    infl = pd.DataFrame()
    E, B, F = "RelMSE_EN-36", "RelMSE_EBM-24", "RelMSE_fastai-41"

    # ---- H1 (primary, intersection-union, outside Holm)
    if all(c in nm for c in (E, B, F)):
        d = nm.dropna(subset=[E, B, F, "C"]).reset_index(drop=True)
        s = d["subset"].to_numpy()
        lv = [x for x in levels_all if x in set(s)]
        dEN = (d[E] - d[F]).to_numpy()
        dEB = (d[B] - d[F]).to_numpy()
        C = d["C"].to_numpy()
        obs = inf.h1_slopes(dEN, dEB, C, s, lv)
        if inf.min_per_subset(s) < 2:
            obs = (np.nan, np.nan)
        idx = inf.stratified_indices(s, n_boot, rng)
        bt = np.array([inf.h1_slopes(dEN[i], dEB[i], C[i], s[i], lv) for i in idx])
        cnt = d["subset"].value_counts().to_dict()
        ci_en = inf.percentile_ci(bt[:, 0])
        ci_eb = inf.percentile_ci(bt[:, 1])
        confirmed = bool(ci_en[0] > 0 and ci_eb[0] > 0)
        rows.append(_row(cohort, "H1", "slope Delta_EN-fastai on C", obs[0], ci_en[0], ci_en[1], np.nan, cnt,
                         n_boot_valid=int(np.isfinite(bt[:, 0]).sum())))
        rows.append(_row(cohort, "H1", "slope Delta_EBM-fastai on C", obs[1], ci_eb[0], ci_eb[1], np.nan, cnt,
                         n_boot_valid=int(np.isfinite(bt[:, 1]).sum())))
        rows.append(_row(cohort, "H1", "criterion: both 2.5th percentiles > 0", float(confirmed), np.nan,
                         np.nan, np.nan, cnt, confirmed=confirmed,
                         boot_corr=float(pd.Series(bt[:, 0]).corr(pd.Series(bt[:, 1])))))
    else:
        rows.append(_row(cohort, "H1", "not estimable", np.nan, np.nan, np.nan, np.nan, {}, confirmed=False))

    fam = {}
    # ---- H2
    a, b_ = cfg.INFERENCE["h2_contrast"]
    if E in nm and B in nm:
        d = nm.dropna(subset=[E, B, "C"]).reset_index(drop=True)
        s = d["subset"].to_numpy()
        lv = [x for x in levels_all if x in set(s)]
        dd = (d[E] - d[B]).to_numpy()
        C = d["C"].to_numpy()
        cnt = d["subset"].value_counts().to_dict()
        if a in lv and b_ in lv and inf.min_per_subset(s) >= 2:
            sl = inf.h2_slopes(dd, C, s, lv)
            obs = sl[a] - sl[b_]
            idx = inf.stratified_indices(s, n_boot, rng)
            bt = np.array([(lambda z: z[a] - z[b_])(inf.h2_slopes(dd[i], C[i], s[i], lv)) for i in idx])
            lo, hi = inf.percentile_ci(bt)
            p = inf.null_centred_p(bt, obs)
            rows.append(_row(cohort, "H2", "beta_J30 - beta_J120 (Delta_EN-EBM on C)", obs, lo, hi, p, cnt,
                             n_boot_valid=int(np.isfinite(bt).sum())))
            fam["H2"] = len(rows) - 1
            for lvl in lv:
                rows.append(_row(cohort, "H2_slopes", "beta_%s" % lvl, sl[lvl], np.nan, np.nan, np.nan, cnt))
            ts = inf.theilsen_slopes(dd, C, s, lv)
            ts_obs = ts[a] - ts[b_]
            ts_bt = np.array([(lambda z: z[a] - z[b_])(inf.theilsen_slopes(dd[i], C[i], s[i], lv)) for i in idx])
            tlo, thi = inf.percentile_ci(ts_bt)
            robust = bool(np.sign(ts_obs) == np.sign(obs) and abs(ts_obs) >= 0.5 * abs(obs))
            rows.append(_row(cohort, "H2_sensitivity", "Theil-Sen (separate) J30 - J120", ts_obs, tlo, thi,
                             np.nan, cnt, robust_reading=robust))
            X = inf._design(C, s, lv, True)
            w = inf.subset_weights(s)
            D = inf.cooks_distance(dd, X, w)
            loo = []
            for i in range(len(d)):
                m = np.ones(len(d), bool)
                m[i] = False
                z = inf.h2_slopes(dd[m], C[m], s[m], lv)
                loo.append(z[a] - z[b_] - obs)
            infl = pd.DataFrame({"cohort": cohort, "network_id": d["network_id"], "subset": s,
                                 "cooks_D": D, "flag_D_gt_4_over_n": D > 4.0 / len(d),
                                 "loo_contrast_change": loo})
        else:
            rows.append(_row(cohort, "H2", "not estimable (missing J30/J120 or < 2 networks in a subset)", np.nan, np.nan, np.nan,
                             np.nan, cnt))
            fam["H2"] = len(rows) - 1

    # ---- H3 / H4: weighted paired means
    def paired(hyp, col_a, col_b, statistic, extra_mask=None):
        if col_a not in nm or col_b not in nm:
            rows.append(_row(cohort, hyp, statistic + " (not estimable)", np.nan, np.nan, np.nan, np.nan, {}))
            fam[hyp] = len(rows) - 1
            return
        d = nm.dropna(subset=[col_a, col_b])
        if extra_mask is not None:
            d = d[extra_mask(d)]
        d = d.reset_index(drop=True)
        cnt = d["subset"].value_counts().to_dict()
        if len(d) == 0 or inf.min_per_subset(d["subset"]) < 2:
            rows.append(_row(cohort, hyp, statistic + " (not estimable: < 2 networks in a subset)", np.nan, np.nan,
                             np.nan, np.nan, cnt))
            fam[hyp] = len(rows) - 1
            return
        diff = (d[col_a] - d[col_b]).to_numpy()
        s = d["subset"].to_numpy()
        obs = inf.weighted_subset_mean(diff, s)
        idx = inf.stratified_indices(s, n_boot, rng)
        bt = np.array([inf.weighted_subset_mean(diff[i], s[i]) for i in idx])
        lo, hi = inf.percentile_ci(bt)
        desc = inf.describe_pair(diff, s)
        rows.append(_row(cohort, hyp, statistic, obs, lo, hi, inf.null_centred_p(bt, obs), cnt,
                         median=desc["median"], win_EN=desc["win_first"], win_EBM=desc["win_second"]))
        fam[hyp] = len(rows) - 1

    paired("H3", "MAE_D_EN-36", "MAE_D_EBM-24", "mean EN-EBM MAE_D")
    k = cfg.DECISION["h4_k"]
    paired("H4", "regret%d_EN-36" % k, "regret%d_EBM-24" % k, "mean EN-EBM regret (k=3, duration units)",
           extra_mask=lambda d: d["n_eligible"] >= k)
    # ---- Holm over the predefined family; non-estimable -> p = 1
    fam_ids = cfg.INFERENCE["holm_family"]
    ps = [rows[fam[h]]["p_raw"] if h in fam else np.nan for h in fam_ids]
    adj = inf.holm(ps)
    for h, pa in zip(fam_ids, adj):
        if h in fam:
            rows[fam[h]]["p_holm"] = pa
            rows[fam[h]]["confirmed"] = bool(np.isfinite(rows[fam[h]]["p_raw"]) and pa <= cfg.INFERENCE["family_alpha"])
    # ---- descriptive pairs (always mean, median and win frequency together)
    pairs = [("EN-36", "EBM-24"), ("EN-36", "fastai-41"), ("EBM-24", "fastai-41"), ("OLS", "EN-36"),
             ("OLS", "EBM-24"), ("EBM-24", "EBM-24-noint"), ("EN-36", "EN-36-main"), ("EN-36", "EN-36-centred")]
    for m1, m2 in pairs:
        c1, c2 = "RelMSE_" + m1, "RelMSE_" + m2
        if c1 in nm and c2 in nm:
            d = nm.dropna(subset=[c1, c2])
            desc = inf.describe_pair((d[c1] - d[c2]).to_numpy(), d["subset"].to_numpy())
            rows.append(_row(cohort, "descriptive", "RelMSE %s - %s" % (m1, m2), desc["weighted_mean"], np.nan,
                             np.nan, np.nan, d["subset"].value_counts().to_dict(), median=desc["median"],
                             win_first=desc["win_first"], win_second=desc["win_second"], ties=desc["ties"]))
    if "baseline_mean_eval" in nm and ("regret%d_EN-36" % k) in nm and ("regret%d_EBM-24" % k) in nm:
        d = nm.dropna(subset=["regret%d_EN-36" % k, "regret%d_EBM-24" % k])
        d = d[d["n_eligible"] >= k]
        if len(d):
            v = ((d["regret%d_EN-36" % k] - d["regret%d_EBM-24" % k]) / d["baseline_mean_eval"]).to_numpy()
            rows.append(_row(cohort, "descriptive", "H4 scaled by baseline mean duration",
                             inf.weighted_subset_mean(v, d["subset"].to_numpy()), np.nan, np.nan, np.nan,
                             d["subset"].value_counts().to_dict()))
    return pd.DataFrame(rows), infl


def run(ctx, n_boot=None):
    tabs = collect(ctx)
    out = os.path.join(ctx.cohort_dir, "aggregate")
    common.ensure_dir(out)
    for name, df in tabs.items():
        if df.empty:
            continue
        if name == "sensitivity":
            common.write_table(df, os.path.join(out, name))
        else:
            common.atomic_write_csv(df, os.path.join(out, name + ".csv"))
    nm = network_metrics(tabs)
    common.atomic_write_csv(nm, os.path.join(out, "network_metrics.csv"))
    infr, infl = run_inference(ctx.cohort, nm, n_boot=n_boot)
    common.atomic_write_csv(infr, os.path.join(out, "inference.csv"))
    if not infl.empty:
        common.atomic_write_csv(infl, os.path.join(out, "h2_influence.csv"))
    write_run_manifest(ctx, n_boot or cfg.INFERENCE["n_boot"])
    return infr


def write_run_manifest(ctx, n_boot):
    stamps = _current(glob.glob(os.path.join(ctx.cohort_dir, "networks", "*", "*", "_stamp.json")) +
                      glob.glob(os.path.join(ctx.cohort_dir, "networks", "*", "*", "*", "_stamp.json")))
    status = {}
    times = []
    for f in stamps:
        with open(f) as fh:
            r = json.load(fh)
        rel = os.path.relpath(os.path.dirname(f), os.path.join(ctx.cohort_dir, "networks")).split(os.sep)
        key = "/".join(rel[1:])
        status.setdefault(key, {}).setdefault(r.get("status", "?"), 0)
        status[key][r.get("status", "?")] += 1
        times.append(r.get("finished_utc"))
    times = sorted(t for t in times if t)
    man = {"protocol_version": cfg.PROTOCOL_VERSION, "code_commit": common.git_commit(ctx.repo),
           "config_hash": cfg.config_hash(), "code_hash": common.code_hash(),
           "environment": common.environment_info(), "start": times[0] if times else None,
           "end": common.utcnow(), "stage_status": status, "n_boot": n_boot,
           "deviations": DEVIATIONS, "cohort": ctx.cohort}
    common.atomic_write_json(os.path.join(ctx.cohort_dir, "run_manifest.json"), man)
    lines = ["# run_manifest (%s)" % ctx.cohort, ""]
    for k in ("protocol_version", "code_commit", "config_hash", "code_hash", "start", "end", "n_boot"):
        lines.append("- %s: %s" % (k, man[k]))
    lines += ["", "## environment", ""] + ["- %s: %s" % (k, v) for k, v in sorted(man["environment"].items())]
    lines += ["", "## stage_status", "", "| stage | status counts |", "| --- | --- |"]
    lines += ["| %s | %s |" % (k, ", ".join("%s=%d" % kv for kv in sorted(v.items()))) for k, v in sorted(status.items())]
    lines += ["", "## deviations", ""] + ["- " + d for d in DEVIATIONS]
    common.atomic_write_text(os.path.join(ctx.cohort_dir, "run_manifest.md"), "\n".join(lines) + "\n")
