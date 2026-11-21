#!/usr/bin/python
"""OLS reference model for the Section 5 benchmark and seven-model extension.

OLS is added as a linear reference without retuning or changing the six retained
configurations.

Subcommands
-----------
fit      Fit OLS per evaluation network on the ORIGINAL scenarios
         (montecarlo/<p>_simulation_01.csv) and the global split
         montecarlo/test_runs.csv. One CSV per network is written to
         <out>/per_network/ (resumable), then merged to <out>/ols_relmse_360.csv.
analyse  Seven-model (six families + OLS) Friedman test, Conover post-hoc and
         summaries, labelled as an extension. The six-family tables are not touched.

OLS definition
--------------
Inputs: all duration<n> columns that are non-constant in the training scenarios
(constant = nunique <= 1 in training, the same rule as fastai 'fix A' in
model_evaluation.py), standardised with StandardScaler fitted on training only.
Model: sklearn LinearRegression (intercept, no regularisation).
RelMSE = SSE(test) / SSE(test vs mean(y_train)) as in model_evaluation.py;
the denominator being zero gives inf, as in model_evaluation.py.
time = wall-clock seconds from the start of preprocessing to RelMSE (same span as
model_evaluation.py); 'time' is rounded to whole seconds as in the benchmark,
'time_raw' keeps the unrounded value. One BLAS/OpenMP thread per process.

Statistical procedure (copied from model_evaluation.ipynb)
----------------------------------------------------------
Friedman: scipy.stats.friedmanchisquare on the complete network x model matrix.
Post-hoc: scikit_posthocs.posthoc_conover_friedman(matrix, p_adjust='bonferroni'),
alpha = 0.05; ranks: method='average', lower RelMSE = better. With 7 models the
Bonferroni factor is 21 pairs (15 in the six-family analysis). Because several
p-values underflow double precision, log10 p-values are also reported, computed
with the same Conover t statistic and df = n*k - k - n + 1 (mpmath for the extreme
tail, used only in 'analyse').
Bootstrap: 10,000 resamples of networks, seed 112358, percentile 95 % CI. The
notebook resamples the 360 networks without stratification; that is reproduced
('nb' columns). A bootstrap stratified by PSPLIB set is reported
alongside ('strat' columns).

Python 3.9 compatible.
"""
import argparse
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import random  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SEED = 112358
N_BOOTSTRAP = 10000
CI_LEVEL = 0.95
SIX_FAMILIES = {"ebm": 24, "rf": 14, "xgb": 39, "sv": 0, "en": 36, "fastai": 41}
OLS_LABEL = "ols"


def reset_seeds(seed_value):
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)


# ----------------------------------------------------------------------------- fit
def evaluate_ols(repo, project, subset, test_idx):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    from sklearn.preprocessing import StandardScaler

    reset_seeds(SEED)
    df = pd.read_csv(os.path.join(repo, "montecarlo", project + "_simulation_01.csv"), index_col=0)
    df_train = df.loc[df.index.difference(test_idx)]
    df_test = df.loc[test_idx]
    dur_cols = [c for c in df.columns if c.startswith("duration")]

    start = time.time()
    const_cols = [c for c in dur_cols if df_train[c].nunique() <= 1]
    x_cols = [c for c in dur_cols if c not in const_cols]
    y_train = df_train["actual_duration"].to_numpy()
    y_test = df_test["actual_duration"].to_numpy()
    scaler = StandardScaler()
    x_train = scaler.fit_transform(df_train[x_cols].to_numpy())
    x_test = scaler.transform(df_test[x_cols].to_numpy())
    model = LinearRegression(fit_intercept=True)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    mse_base = mean_squared_error(y_test, np.repeat(y_train.mean(), len(y_test)))
    mse_model = mean_squared_error(y_test, pred)
    rel_mse = mse_model / mse_base if mse_base != 0 else np.inf
    elapsed = time.time() - start

    return {
        "project": project,
        "subset": subset,
        "model": OLS_LABEL,
        "n_train": len(df_train),
        "n_test": len(df_test),
        "n_inputs": len(x_cols),
        "n_constant_dropped": len(const_cols),
        "sse_test": float(((y_test - pred) ** 2).sum()),
        "sse_base": float(((y_test - y_train.mean()) ** 2).sum()),
        "RelMSE": rel_mse,
        "time": round(elapsed, 0),
        "time_raw": elapsed,
    }


def _worker(job):
    repo, out_dir, project, subset, test_idx = job
    out = os.path.join(out_dir, "per_network", "ols_" + project + ".csv")
    if os.path.exists(out):
        return out
    row = evaluate_ols(repo, project, subset, test_idx)
    tmp = out + ".tmp"
    pd.DataFrame([row]).to_csv(tmp, index=False)
    os.replace(tmp, out)
    return out


def cmd_fit(args):
    import multiprocessing as mp

    repo = args.repo
    out_dir = args.out
    os.makedirs(os.path.join(out_dir, "per_network"), exist_ok=True)
    pe = pd.read_csv(os.path.join(repo, "model_evaluation", "project_evaluation.csv"))
    pe["project"] = pe["FileName"].astype(str).str.replace(".sm", "", regex=False)
    if args.limit:
        pe = pe.head(args.limit)
    test_idx = pd.read_csv(os.path.join(repo, "montecarlo", "test_runs.csv"))["test_runs"].values
    jobs = [(repo, out_dir, p, s.upper(), test_idx) for p, s in zip(pe["project"], pe["SubSet"])]
    if args.workers > 1:
        with mp.get_context("spawn").Pool(args.workers) as pool:
            for i, _ in enumerate(pool.imap_unordered(_worker, jobs), 1):
                if i % 40 == 0:
                    print("done", i, "of", len(jobs), flush=True)
    else:
        for job in jobs:
            _worker(job)
    files = [os.path.join(out_dir, "per_network", "ols_" + p + ".csv") for p in pe["project"]]
    merged = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    merged.to_csv(os.path.join(out_dir, "ols_relmse_360.csv"), index=False)
    print("wrote", len(merged), "rows")


# ------------------------------------------------------------------------- analyse
def load_six(repo):
    rd = os.path.join(repo, "model_evaluation", "results")
    files = sorted(f for f in os.listdir(rd) if f.startswith("model_evaluation_") and f.endswith(".csv"))
    res = pd.concat([pd.read_csv(os.path.join(rd, f), usecols=["project", "subset", "model", "lhs", "RelMSE", "time"])
                     for f in files], ignore_index=True)
    res["subset"] = res["subset"].astype(str).str.upper()
    res["model_label"] = res["model"] + "-" + res["model"].map(SIX_FAMILIES).astype(str)
    return res


def percentile_ci(values):
    a = 1 - CI_LEVEL
    return np.quantile(values, [a / 2, 1 - a / 2], axis=0)


def bootstrap_indices(subsets, stratified, seed):
    rng = np.random.default_rng(seed)
    n = len(subsets)
    if not stratified:
        return rng.integers(0, n, size=(N_BOOTSTRAP, n))
    parts = []
    for s in sorted(np.unique(subsets)):
        pos = np.flatnonzero(subsets == s)
        parts.append(pos[rng.integers(0, len(pos), size=(N_BOOTSTRAP, len(pos)))])
    return np.concatenate(parts, axis=1)


def _t_logsf_mp(t, df):
    """log survival function of Student t for extreme t, via mpmath if available:
    sf = 0.5 * I_x(df/2, 1/2), x = df / (df + t^2). Returns -inf if mpmath is absent."""
    try:
        import mpmath
    except ImportError:
        return -np.inf
    mpmath.mp.dps = 50
    x = mpmath.mpf(df) / (mpmath.mpf(df) + mpmath.mpf(t) ** 2)
    sf = 0.5 * mpmath.betainc(mpmath.mpf(df) / 2, 0.5, 0, x, regularized=True)
    return float(mpmath.log(sf))


def conover_log10_unadjusted(data):
    """Conover-Friedman t statistics exactly as in scikit-posthocs 0.17.1
    (posthoc_conover_friedman, complete block matrix branch), returning log10 of
    the two-sided unadjusted p-value so that p-values below double precision can
    be reported. Bonferroni-adjusted log10 p = min(0, log10 p + log10(#pairs))."""
    from scipy import stats
    values = data.to_numpy()
    n, k = values.shape
    ranks = stats.rankdata(values, axis=1, method="average")
    rank_sums = ranks.sum(axis=0)
    a1 = np.sum(ranks ** 2.0)
    s2 = 1.0 / (k - 1.0) * (a1 - k * n * ((k + 1.0) ** 2.0) / 4.0)
    t2 = np.sum((rank_sums - n * (k + 1.0) / 2.0) ** 2.0) / s2
    df = n * k - k - n + 1.0
    aa = s2 * (2.0 * n * (k - 1.0)) / df
    bb = 1.0 - t2 / (n * (k - 1.0))
    out = np.zeros((k, k))
    tvals = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            t = abs(rank_sums[i] - rank_sums[j]) / np.sqrt(aa) / np.sqrt(bb)
            tvals[i, j] = t
            lsf = stats.t.logsf(t, df)
            if not np.isfinite(lsf):
                lsf = _t_logsf_mp(t, df)
            out[i, j] = (np.log(2.0) + lsf) / np.log(10.0)
    return (pd.DataFrame(out, index=data.columns, columns=data.columns),
            pd.DataFrame(tvals, index=data.columns, columns=data.columns), df)


def cmd_analyse(args):
    from itertools import combinations
    from scipy import stats
    import scikit_posthocs as sp

    repo = args.repo
    out_dir = args.out
    six = load_six(repo)
    ols = pd.read_csv(args.ols_csv)
    order = [m + "-" + str(SIX_FAMILIES[m]) for m in ["ebm", "rf", "xgb", "sv", "en", "fastai"]] + [OLS_LABEL]
    rel = six.pivot(index="project", columns="model_label", values="RelMSE")
    tim = six.pivot(index="project", columns="model_label", values="time")
    rel[OLS_LABEL] = ols.set_index("project").loc[rel.index, "RelMSE"]
    tim[OLS_LABEL] = ols.set_index("project").loc[rel.index, "time"]
    rel = rel[order]
    tim = tim[order]
    subsets = six[["project", "subset"]].drop_duplicates().set_index("project").loc[rel.index, "subset"].to_numpy()
    clean = rel.dropna()
    assert len(clean) == 360, "incomplete seven-model matrix"

    ranks = clean.rank(axis=1, ascending=True, method="average")
    avg_rank = ranks.mean().sort_values()
    fr = stats.friedmanchisquare(*[clean[c].to_numpy() for c in clean.columns])
    k = clean.shape[1]
    log10_p = stats.chi2.logsf(fr.statistic, k - 1) / np.log(10.0)
    conover = sp.posthoc_conover_friedman(clean, p_adjust="bonferroni").loc[avg_rank.index, avg_rank.index]
    lg, tv, cdf = conover_log10_unadjusted(clean)
    m_pairs = k * (k - 1) / 2.0
    best = avg_rank.index[0]
    best_group = [m for m in avg_rank.index if conover.loc[best, m] >= 0.05]

    pd.DataFrame([{
        "analysis": "seven-model extension (six families + OLS); not the original six-family test",
        "n_projects": len(clean), "n_models": k, "friedman_statistic": fr.statistic,
        "friedman_pvalue": fr.pvalue, "friedman_log10_pvalue": log10_p,
        "best_model": best, "best_group": ", ".join(best_group),
        "posthoc": "scikit_posthocs.posthoc_conover_friedman", "p_adjust": "bonferroni",
        "n_pairs": int(m_pairs), "alpha": 0.05,
        "scikit_posthocs_version": sp.__version__,
    }]).to_csv(os.path.join(out_dir, "seven_model_friedman.csv"), index=False)
    conover.to_csv(os.path.join(out_dir, "seven_model_conover_all.csv"))

    rows = []
    for a, b in combinations(list(avg_rank.index), 2):
        l10 = lg.loc[a, b]
        l10_adj = min(0.0, l10 + np.log10(m_pairs))
        rows.append({"model_a": a, "model_b": b, "mean_rank_a": avg_rank[a], "mean_rank_b": avg_rank[b],
                     "conover_t": tv.loc[a, b], "df": cdf,
                     "p_bonferroni_scikit_posthocs": conover.loc[a, b],
                     "log10_p_unadjusted": l10, "log10_p_bonferroni": l10_adj,
                     "significant_0.05": bool(conover.loc[a, b] < 0.05),
                     "friedman_statistic": fr.statistic, "friedman_pvalue": fr.pvalue,
                     "friedman_log10_pvalue": log10_p, "n_projects": len(clean),
                     "analysis": "seven-model extension"})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "seven_model_friedman_conover.csv"), index=False)

    # summaries with bootstrap CIs (notebook-style unstratified and subset-stratified)
    vals = clean.to_numpy()
    rvals = ranks.to_numpy()
    wins = clean.idxmin(axis=1).value_counts()
    stats_rows = {m: {"model": m, "n_projects": len(clean), "mean_rank": ranks[m].mean(),
                      "median_rank": ranks[m].median(), "wins": int(wins.get(m, 0)),
                      "mean_RelMSE": clean[m].mean(), "median_RelMSE": clean[m].median(),
                      "p95_RelMSE": float(np.quantile(clean[m], 0.95)), "max_RelMSE": clean[m].max(),
                      "std_RelMSE": clean[m].std(),
                      "mean_time": tim.loc[clean.index, m].mean(), "median_time": tim.loc[clean.index, m].median()}
                  for m in clean.columns}
    for tag, strat in (("nb", False), ("strat", True)):
        idx = bootstrap_indices(subsets, strat, SEED)
        for j, m in enumerate(clean.columns):
            s = vals[idx, j]
            for name, fn in (("mean_RelMSE", lambda z: z.mean(axis=1)),
                             ("median_RelMSE", lambda z: np.median(z, axis=1)),
                             ("p95_RelMSE", lambda z: np.quantile(z, 0.95, axis=1))):
                lo, hi = percentile_ci(fn(s))
                stats_rows[m][name + "_ci_low_" + tag] = lo
                stats_rows[m][name + "_ci_high_" + tag] = hi
            lo, hi = percentile_ci(rvals[idx, j].mean(axis=1))
            stats_rows[m]["mean_rank_ci_low_" + tag] = lo
            stats_rows[m]["mean_rank_ci_high_" + tag] = hi
    summ = pd.DataFrame([stats_rows[m] for m in avg_rank.index])
    summ.to_csv(os.path.join(out_dir, "seven_model_summary_all.csv"), index=False)

    # by-subset mean rank / median / wins (descriptive)
    srows = []
    for s in ["J30", "J60", "J90", "J120"]:
        sub = clean.loc[subsets == s]
        rk = sub.rank(axis=1, method="average").mean()
        w = sub.idxmin(axis=1).value_counts()
        frs = stats.friedmanchisquare(*[sub[c].to_numpy() for c in sub.columns])
        for m in clean.columns:
            srows.append({"subset": s, "model": m, "n_projects": len(sub), "mean_rank": rk[m],
                          "median_RelMSE": sub[m].median(), "mean_RelMSE": sub[m].mean(),
                          "wins": int(w.get(m, 0)), "friedman_pvalue_subset": frs.pvalue})
    pd.DataFrame(srows).to_csv(os.path.join(out_dir, "seven_model_summary_by_subset.csv"), index=False)
    print(summ[["model", "mean_rank", "wins", "median_RelMSE", "mean_RelMSE", "p95_RelMSE", "max_RelMSE"]].to_string())
    print("Friedman", fr.statistic, fr.pvalue, "log10p", log10_p, "best group", best_group)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["fit", "analyse"])
    ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--out", default="ols_out")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="fit only the first N networks (testing)")
    ap.add_argument("--ols-csv", default="ols_out/ols_relmse_360.csv")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    if args.command == "fit":
        cmd_fit(args)
    else:
        cmd_analyse(args)


if __name__ == "__main__":
    main()
