"""Acceptance checks of protocol §7.2 for the Sections 6-7 pipeline.

Unit checks (Holm, null-centred p, seeds, analytic S1/ST, row-order invariance) need no data.
Data checks read the outputs of a local run of the pipeline:

    python -m scripts.sections67 prepare   --cohort localtest --network-list nets.csv --results R
    python -m scripts.sections67 simulate  --cohort localtest --network all --results R
    python -m scripts.sections67 structure --cohort localtest --network all --results R
    python -m scripts.sections67 fit       --cohort localtest --network all --results R
    python -m scripts.sections67 explain   --cohort localtest --network all --results R
    python -m scripts.sections67 decide    --cohort localtest --network all --results R
    SECTIONS67_TEST_RESULTS=R pytest scripts/sections67/tests -v

Environment variables: SECTIONS67_TEST_RESULTS (results dir), SECTIONS67_TEST_COHORT (default
'localtest'), SECTIONS67_C_REFERENCE (optional CSV with project,C reference values),
SECTIONS67_TEST_REPORT (optional CSV to which every check appends name, value, pass).
Data checks are skipped when the results directory is missing.
"""
import csv
import glob
import hashlib
import os
import types

import numpy as np
import pandas as pd
import pytest

from scripts.sections67 import common, data, decide, explain, inference, models, simulate
from scripts.sections67 import config as cfg
from scripts.sections67.network import critical_matrix, critical_set_stats

RESULTS = os.environ.get("SECTIONS67_TEST_RESULTS")
COHORT = os.environ.get("SECTIONS67_TEST_COHORT", "localtest")
REPO = data.default_repo()
needs_results = pytest.mark.skipif(not RESULTS or not os.path.isdir(os.path.join(RESULTS or "", COHORT)),
                                   reason="no local pipeline results (set SECTIONS67_TEST_RESULTS)")


def record(name, value, passed, detail=""):
    path = os.environ.get("SECTIONS67_TEST_REPORT")
    if path:
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["check", "value", "pass", "detail"])
            w.writerow([name, value, bool(passed), detail])
    assert passed, "%s: %s %s" % (name, value, detail)


def ctx():
    return data.Ctx(REPO, RESULTS, COHORT)


def networks():
    if not RESULTS:
        return []
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(RESULTS, COHORT, "networks", "*"))
                  if ".superseded-" not in p)


# ====================================================================== unit checks
def test_derive_seed_rule():
    text = "20261121|j301_1|simulate|0"
    expect = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little") % 2 ** 32
    s = common.derive_seed("j301_1", "simulate", 0)
    record("seed rule SHA-256 first 8 bytes LE mod 2**32", s, s == expect)
    assert common.derive_seed("j301_1", "simulate", 1) != s
    with pytest.raises(ValueError):
        common.derive_seed("j301_1", "not_a_stage", 0)


def test_holm_known_example():
    adj = inference.holm([0.01, 0.04, 0.03])
    ok1 = np.allclose(adj, [0.03, 0.06, 0.06])
    adj2 = inference.holm([0.01, 0.02, 0.03, 0.04, 0.05])
    ok2 = np.allclose(adj2, [0.05, 0.08, 0.09, 0.09, 0.09])
    adj3 = inference.holm([0.5, np.nan, 0.01])          # non-estimable -> p = 1
    ok3 = np.allclose(adj3, [1.0, 1.0, 0.03])
    record("Holm known examples", "%s | %s | %s" % (np.round(adj, 4).tolist(), np.round(adj2, 4).tolist(),
                                                     np.round(adj3, 4).tolist()), ok1 and ok2 and ok3)


def test_null_centred_p():
    boot = np.array([1.0, 2.0, 3.0, 4.0])
    p = inference.null_centred_p(boot, 2.0)       # |b-2| >= 2 only for b=4 -> (1+1)/(4+1)
    record("null-centred bootstrap p example", p, abs(p - 0.4) < 1e-12)


class _LinearDesign:
    """Minimal stand-in for explain.Design on independent uniform inputs."""

    def __init__(self, n=20000, p=5, seed=1):
        rng = np.random.default_rng(seed)
        self.variable = np.arange(p)
        self.steps = np.array(cfg.EXPLAIN["steps_sd"])
        self.S = rng.uniform(0, 1, size=(n, p))
        self.X0 = self.S[:200]
        self.delta = np.ones((p, len(self.steps), 200)) * self.steps[None, :, None] * 0.1
        q = np.linspace(0.01, 0.99, 21)
        self.pd_bg = self.S[:500]
        self.pd_grid = [np.unique(np.quantile(self.S[:, j], q)) for j in range(p)]
        self.s1_idx = np.arange(500)
        self.s1_grid = self.pd_grid
        self.perms = [[np.random.default_rng(100 * r + j).permutation(n) for j in range(p)] for r in range(3)]

    def perturbed_makers(self):
        out = []
        for a in range(len(self.variable)):
            for s in range(len(self.steps)):
                def mk(a=a, s=s):
                    X = self.X0.copy()
                    X[:, a] += self.delta[a, s]
                    return X
                out.append(mk)
        return out


def test_linear_function_S1_ST_analytic():
    b = np.array([1.0, 0.5, 0.0, 2.0, 0.0])
    dz = _LinearDesign()
    f = lambda X: X @ b + 3.0                      # noqa: E731
    res = explain.function_outputs(f, dz)
    var_x = 1.0 / 12.0
    analytic = b ** 2 * var_x / np.sum(b ** 2 * var_x)
    err_s1 = float(np.max(np.abs(res["S1"] - analytic)))
    err_st = float(np.max(np.abs(res["ST"].mean(1) - analytic)))
    irrelevant = float(np.max(np.abs(np.r_[res["S1"][b == 0], res["ST"][b == 0].ravel()])))
    record("analytic S1 on linear function (max abs err)", err_s1, err_s1 < 0.03)
    record("analytic ST on linear function (max abs err)", err_st, err_st < 0.03)
    record("irrelevant activities S1=ST=0 (max abs; BLAS rounding only)", irrelevant, irrelevant < 1e-15)
    D = res["dy"] / dz.delta
    record("linear function local D equals coefficient (max abs err)",
           float(np.max(np.abs(D - b[:, None, None]))), np.allclose(D, b[:, None, None], atol=1e-9))


def test_row_order_invariance_unit():
    rng = np.random.default_rng(3)
    crit = rng.random((2000, 12)) < 0.3
    perm = rng.permutation(2000)
    a, _, _ = critical_set_stats(crit)
    b, _, _ = critical_set_stats(crit[perm])
    vals = rng.random((4, 4, 300))
    valid = rng.random((4, 4, 300)) > 0.2
    p2 = rng.permutation(300)
    m1 = explain.hier_mean(vals, valid)[0]
    m2 = explain.hier_mean(vals[:, :, p2], valid[:, :, p2])[0]
    n = 40
    nm = pd.DataFrame({"subset": np.repeat(["J30", "J60", "J90", "J120"], n // 4), "C": rng.random(n),
                       "d1": rng.normal(size=n), "d2": rng.normal(size=n)})
    lv = cfg.COHORTS["subsets"]
    s1 = inference.h1_slopes(nm.d1, nm.d2, nm.C, nm.subset, lv)
    q = nm.sample(frac=1, random_state=5)
    s2 = inference.h1_slopes(q.d1, q.d2, q.C, q.subset, lv)
    h2a = inference.h2_slopes(nm.d1, nm.C, nm.subset, lv)
    h2b = inference.h2_slopes(q.d1, q.C, q.subset, lv)
    w1 = inference.weighted_subset_mean(nm.d1, nm.subset)
    w2 = inference.weighted_subset_mean(q.d1, q.subset)
    ok = (a == b and abs(m1 - m2) < 1e-12 and np.allclose(s1, s2, atol=1e-12)
          and all(abs(h2a[k] - h2b[k]) < 1e-12 for k in lv) and abs(w1 - w2) < 1e-12)
    record("row-order invariance (C stats, local means, H1/H2 slopes, H3/H4 mean)", ok, ok)


def test_cap_never_increases_unit():
    rng = np.random.default_rng(0)
    X = rng.random((500, 6))
    med = np.median(X, 0)
    Xc = decide.cap(X, [1, 3], med)
    ok = bool((Xc <= X).all() and (Xc[:, [0, 2, 4, 5]] == X[:, [0, 2, 4, 5]]).all())
    record("cap is componentwise non-increasing and only touches chosen columns", ok, ok)


# ====================================================================== data checks
@needs_results
def test_splits_and_identifiers():
    c = ctx()
    test = set(int(i) for i in c.test_runs)
    ok = True
    for n in networks():
        for meta_path in glob.glob(os.path.join(c.stage_dir(n, "fit"), "*", "model_meta.json")):
            import json
            with open(meta_path) as f:
                meta = json.load(f)
            ok &= set(meta["test_indices"]) == test and len(meta["test_indices"]) == 2000
            df, dcols, _ = data.load_scenarios(c, n)
            ok &= meta["column_order"] == dcols
            tr, te = data.split(df, c.test_runs)
            ok &= len(set(tr.index) & set(te.index)) == 0 and len(tr) + len(te) == len(df)
    record("partitions and column order match test_runs.csv, no train/test overlap", ok, ok)


@needs_results
def test_reload_reproducibility():
    c = ctx()
    worst = 0.0
    n_models = 0
    for n in networks():
        df, dcols, _ = data.load_scenarios(c, n)
        X = df.loc[c.test_runs, dcols].to_numpy(float)
        for d in sorted(x for x in glob.glob(os.path.join(c.stage_dir(n, "fit"), "*")) if ".superseded-" not in x):
            m = os.path.basename(d)
            if not os.path.exists(os.path.join(d, "model_meta.json")):
                continue
            pipe, _ = models.load(d, m)
            pred = common.read_table(os.path.join(d, "predictions"))
            pred = pred.set_index("scenario_id").loc[c.test_runs, "y_pred"].to_numpy(float)
            worst = max(worst, float(np.max(np.abs(pipe.predict(X) - pred))))
            n_models += 1
    record("reload reproduces stored predictions (max abs diff, %d fitted models)" % n_models, worst,
           n_models > 0 and worst <= 1e-9)


@needs_results
def test_en_relmse_matches_benchmark():
    c = ctx()
    rows = []
    for n in networks():
        f = os.path.join(c.stage_dir(n, "fit", "EN-36"), "fit_manifest.csv")
        if os.path.exists(f):
            from scripts.sections67 import fit_stage
            r = pd.read_csv(f, float_precision="round_trip").iloc[0]
            rows.append((n, float(r["RelMSE"]), fit_stage.stored_benchmark_relmse(c, n, "EN-36")))
    rel = [abs(a / b - 1) for _, a, b in rows]
    exact = sum(a == b for _, a, b in rows)
    record("EN-36 refit RelMSE vs stored benchmark (max rel diff; %d/%d bit-identical)" % (exact, len(rows)),
           max(rel) if rel else np.nan, len(rows) > 0 and max(rel) <= 1e-12,
           "; ".join("%s %r vs %r" % t for t in rows))


@needs_results
def test_exact_D_in_unit_interval_and_monotone():
    c = ctx()
    lo, hi, neg = np.inf, -np.inf, 0.0
    for n in networks():
        z = np.load(os.path.join(c.stage_dir(n, "explain", "exact"), "exact_reference.npz"))
        dz = explain.Design(c, n, need_sobol=False)
        D = z["dy"] / dz.delta
        v = dz.valid
        lo, hi = min(lo, D[v].min()), max(hi, D[v].max())
        pos = dz.delta > 0
        neg = min(neg, float(z["dy"][pos].min()))
    record("exact D within [0,1] (min,max over valid perturbations)", "%.3g, %.6g" % (lo, hi),
           lo >= -1e-9 and hi <= 1 + 1e-9)
    record("g does not decrease when an activity is lengthened (min dy)", neg, neg >= -1e-9)


@needs_results
def test_cap_never_increases_g():
    c = ctx()
    worst = -np.inf
    for n in networks():
        for f in glob.glob(os.path.join(c.stage_dir(n, "decide"), "*", "decisions.csv")):
            worst = max(worst, pd.read_csv(f)["cap_max_increase"].max())
        dd = decide.DecisionData(c, n)
        rng = np.random.default_rng(7)
        for _ in range(20):
            cols = rng.choice(dd.eligible, size=min(3, len(dd.eligible)), replace=False)
            inc = dd.net.makespan(decide.cap(dd.Xev, list(cols), dd.med)) - dd.net.makespan(dd.Xev)
            worst = max(worst, float(inc.max()))
    record("median cap never increases g scenario by scenario (max increase)", worst, worst <= 1e-9)


@needs_results
def test_exhaustive_not_worse_than_greedy():
    c = ctx()
    worst = -np.inf
    n_cmp = 0
    for n in networks():
        f = os.path.join(c.stage_dir(n, "decide", "exact"), "decisions.csv")
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f)
        ex = d[d.policy == "exhaustive"].set_index("k")["mean_g_select"]
        gr = d[d.policy == "exact_greedy"].set_index("k")["mean_g_select"]
        for k in ex.index:
            worst = max(worst, ex[k] - gr[k])
            n_cmp += 1
    if n_cmp == 0:
        pytest.skip("no exhaustive search in these networks")
    record("exhaustive <= greedy on selection sample (max of mean_g_ex - mean_g_greedy, %d cmp)" % n_cmp,
           worst, worst <= 1e-9)


@needs_results
def test_C_matches_reference():
    ref_path = os.environ.get("SECTIONS67_C_REFERENCE")
    if not ref_path:
        pytest.skip("SECTIONS67_C_REFERENCE not set")
    ref = pd.read_csv(ref_path).set_index("project")["C"]
    c = ctx()
    diffs = []
    for n in networks():
        f = os.path.join(c.stage_dir(n, "structure"), "structure_metrics.csv")
        if os.path.exists(f) and n in ref.index:
            diffs.append(abs(pd.read_csv(f)["C"].iloc[0] - ref[n]))
    record("C equals reference (max abs diff, %d networks)" % len(diffs), max(diffs) if diffs else np.nan,
           len(diffs) > 0 and max(diffs) < 1e-12)


@needs_results
def test_row_order_invariance_structure():
    c = ctx()
    n = networks()[0]
    b = data.network_bundle(c, n)
    D = b["df"][b["dcols"]].to_numpy(float)
    perm = np.random.default_rng(11).permutation(len(D))
    TF, T = b["net"].floats(D)
    TFp, Tp = b["net"].floats(D[perm])
    s1, _, _ = critical_set_stats(critical_matrix(TF, T, 1e-9))
    s2, _, _ = critical_set_stats(critical_matrix(TFp, Tp, 1e-9))
    ok = s1 == s2 and np.array_equal(T[perm], Tp)
    record("row-order invariance of g and C on scenarios (%s)" % n, ok, ok)


@needs_results
def test_simulator_distributional_equivalence():
    from scipy import stats
    c = ctx()
    n = networks()[0]
    G, sup = data.control_point(c, n)
    nodes = list(G.nodes())
    Dv = simulate.sample_durations(G, nodes, 3000, np.random.default_rng(1))
    Dl = simulate.sample_durations_loop(c, G.copy(), 3000, np.random.default_rng(2))
    df0, dcols, _ = data.load_scenarios(c, n)
    D0 = df0[["duration%s" % i for i in nodes]].to_numpy(float)
    var = np.ptp(D0, axis=0) > 0
    p_loop = min(stats.ks_2samp(Dv[:, j], Dl[:, j]).pvalue for j in np.where(var)[0])
    p_orig = min(stats.ks_2samp(Dv[:, j], D0[:, j]).pvalue for j in np.where(var)[0])
    # original CSV is parsed with pandas' default float parser (as model_evaluation.py): <= 1 ulp
    const_diff = max(float(np.max(np.abs(Dv[:, ~var] - D0[0, ~var]))),
                     float(np.max(np.abs(Dl[:, ~var] - D0[0, ~var]))))
    const_ok = const_diff <= 1e-12
    again = simulate.sample_durations(G, nodes, 3000, np.random.default_rng(1))
    m = int(var.sum())
    thr = 0.001 / m
    record("vectorised vs tools loop sampler: min KS p over %d columns (Bonferroni thr %.1e)" % (m, thr),
           p_loop, p_loop > thr)
    record("vectorised sampler vs original scenarios: min KS p over %d columns" % m, p_orig, p_orig > thr)
    record("constant columns equal original (max abs diff); same seed reproduces draws", const_diff,
           const_ok and np.array_equal(Dv, again))


def test_signs_of_differences_regret_and_guard():
    from scripts.sections67 import aggregate
    rng = np.random.default_rng(4)
    rows, dec = [], []
    for s in cfg.COHORTS["subsets"]:
        for i in range(6):
            nid = "%s_%d" % (s, i)
            C = rng.uniform(0, 0.8)
            rows.append({"cohort": "t", "subset": s, "network_id": nid, "C": C, "OLS_RelMSE": 0.1, "SP": 0.3,
                         "TF": 0.2, "n_variable": 20, "N_eff": 1.0, "CI_share_intermediate": 0.1})
            for pol, b in (("exact_greedy", 1.0), ("EN-36_greedy", 0.7), ("EBM-24_greedy", 0.9)):
                dec.append({"network_id": nid, "policy": pol, "k": 3, "B_mean": b, "n_eligible": 10,
                            "baseline_mean_eval": 50.0})
    tabs = {"structure_metrics": pd.DataFrame(rows), "decisions": pd.DataFrame(dec),
            "fit_manifest": pd.DataFrame([{"network_id": r["network_id"], "model_id": m, "fit_status": "ok",
                                           "RelMSE": v + 0.05 * r["C"] * (m == "EN-36")}
                                          for r in rows for m, v in (("EN-36", 0.03), ("EBM-24", 0.02),
                                                                     ("fastai-41", 0.01))]),
            "local_metrics": pd.DataFrame(), "sensitivity": pd.DataFrame()}
    nm = aggregate.network_metrics(tabs)
    r_en = nm["regret3_EN-36"].unique()
    r_eb = nm["regret3_EBM-24"].unique()
    infr, _ = aggregate.run_inference("t", nm, n_boot=500)
    h4 = infr[infr.hypothesis == "H4"].iloc[0]
    d = infr[infr.statistic == "RelMSE EN-36 - EBM-24"].iloc[0]
    ok = (np.allclose(r_en, 0.3) and np.allclose(r_eb, 0.1) and abs(h4["estimate"] - 0.2) < 1e-12
          and np.isnan(h4["p_raw"]) and d["estimate"] > 0 and d["win_second"] == 1.0)
    record("signs: regret = B_exact - B_model, H4 = regret_EN - regret_EBM, Delta>0 favours 2nd model; "
           "degenerate bootstrap -> p not estimable", ok, ok)
