"""Stage 'decide': median-cap intervention policies (protocol §6.3, §9.4).

x'_i = min(x_i, m_i), m_i = median of the 5,000 selection scenarios.  For an activity in progress
the column is elapsed + (1 - done) * remaining, an increasing affine map of the remaining duration,
so capping the column at its median is the same as capping the remaining duration at its median.
Selection uses only decision_select; every chosen set is evaluated with g on decision_eval.
"""
import itertools
import os
import time

import numpy as np
import pandas as pd

from . import common
from . import config as cfg
from . import data
from . import models
from .explain import batched_eval
from .network import Net, critical_matrix


def cap(X, cols, med):
    Xc = X.copy()
    if len(cols):
        cols = list(cols)
        Xc[:, cols] = np.minimum(X[:, cols], med[cols])
    return Xc


def _cap_maker(X, cols, med):
    return lambda: cap(X, cols, med)


class DecisionData:
    def __init__(self, ctx, network):
        G, support = data.control_point(ctx, network)
        sel, dcols = data.load_extra_sample(ctx, network, "decision_select")
        ev, dc2 = data.load_extra_sample(ctx, network, "decision_eval")
        assert list(dcols) == list(dc2)
        self.dcols = dcols
        self.net = Net(G, dcols)
        self.Xsel = sel[dcols].to_numpy(float)
        self.Xev = ev[dcols].to_numpy(float)
        self.act_ids = np.array([int(c.replace("duration", "")) for c in dcols])
        status = support.set_index("column").loc[dcols, "status"].to_numpy()
        variable = np.ptp(self.Xsel, axis=0) > 0
        self.eligible = np.where(variable & np.isin(status, ["in_progress", "not_started"]))[0]
        self.eligible = self.eligible[np.argsort(self.act_ids[self.eligible], kind="stable")]
        self.med = np.median(self.Xsel, axis=0)
        self.subset = ctx.subset_of(network)


def greedy(f, dd, kmax):
    """Sequential choice maximising the estimated mean reduction; ties -> smallest activity id."""
    chosen = []
    trace = []
    base = float(np.mean(f(dd.Xsel)))
    for _ in range(kmax):
        cands = [j for j in dd.eligible if j not in chosen]
        if not cands:
            break
        vals = batched_eval(f, [_cap_maker(dd.Xsel, chosen + [j], dd.med) for j in cands])
        means = np.array([float(np.mean(v)) for v in vals])
        best = int(np.argmin(means))       # cands are in ascending id order: first = smallest id
        chosen.append(cands[best])
        trace.append({"selected": list(chosen), "est_reduction": base - means[best]})
    return trace


def static_ranking(dd, kind):
    TF, T = dd.net.floats(dd.Xsel)
    CI = critical_matrix(TF, T, cfg.STRUCTURE["critical_tol"]).mean(0)
    if kind == "CI":
        score = CI
    else:
        sdT = T.std()
        score = CI * dd.Xsel.std(0) / sdT if sdT > 0 else np.zeros_like(CI)
    order = sorted(dd.eligible, key=lambda j: (-score[j], dd.act_ids[j]))
    return order, score


def exhaustive(dd, k):
    combos = list(itertools.combinations(list(dd.eligible), k))
    t0 = time.time()
    vals = batched_eval(dd.net.makespan, [_cap_maker(dd.Xsel, list(c), dd.med) for c in combos])
    means = np.array([float(np.mean(v)) for v in vals])
    best = int(np.argmin(means))
    return list(combos[best]), len(combos), time.time() - t0, float(means[best])


def evaluate(dd, cols):
    g_ev = dd.net.makespan(dd.Xev)
    Xc = cap(dd.Xev, cols, dd.med)
    g_c = dd.net.makespan(Xc)
    g_sel_c = dd.net.makespan(cap(dd.Xsel, cols, dd.med))
    return {"B_mean": float(g_ev.mean() - g_c.mean()),
            "B_p90": float(np.percentile(g_ev, 90) - np.percentile(g_c, 90)),
            "expected_duration_reduction": float(np.sum(np.mean(dd.Xev[:, cols] - Xc[:, cols], axis=0)))
            if len(cols) else 0.0,
            "cap_max_increase": float(np.max(g_c - g_ev)),
            "baseline_mean_eval": float(g_ev.mean()),
            "mean_g_select": float(g_sel_c.mean())}


def _row(ctx, network, dd, policy, k, cols, extra=None):
    r = {"cohort": ctx.cohort, "network_id": network, "subset": dd.subset, "policy": policy, "k": k,
         "selected_ids": ";".join(str(int(dd.act_ids[j])) for j in cols), "n_eligible": int(len(dd.eligible))}
    r.update(evaluate(dd, cols))
    r.update(extra or {})
    return r


def run(ctx, network, model_id):
    out = ctx.stage_dir(network, "decide", model_id)
    ins = {}
    for kind in ("decision_select", "decision_eval"):
        base = data.simulated_base(ctx, network, kind)
        ins[kind] = common.sha256_file(base + (".parquet" if os.path.exists(base + ".parquet") else ".csv.gz"))
    if model_id != "exact":
        ins["fit_stamp_sha256"] = common.sha256_file(os.path.join(ctx.stage_dir(network, "fit", model_id),
                                                                  "_stamp.json"))
    st = common.Stamp(out, ins)
    if not st.begin(ctx.force):
        return None
    dd = DecisionData(ctx, network)
    kmax = max(cfg.DECISION["k_values"])
    rows = []
    t0 = time.time()
    if model_id == "exact":
        tr = greedy(dd.net.makespan, dd, kmax)
        for k, step in enumerate(tr, 1):
            rows.append(_row(ctx, network, dd, "exact_greedy", k, step["selected"],
                             {"est_reduction_select": step["est_reduction"]}))
        for kind in cfg.DECISION["policies_static"]:
            order, _ = static_ranking(dd, kind)
            for k in range(1, min(kmax, len(order)) + 1):
                rows.append(_row(ctx, network, dd, kind + "_static", k, order[:k]))
        if dd.subset in cfg.DECISION["exhaustive_subsets"]:
            for k in range(1, min(kmax, len(dd.eligible)) + 1):
                cols, ncomb, secs, mbest = exhaustive(dd, k)
                rows.append(_row(ctx, network, dd, "exhaustive", k, cols,
                                 {"n_combinations": ncomb, "search_time_s": secs}))
    else:
        pipe, meta = models.load(ctx.stage_dir(network, "fit", model_id), model_id)
        assert meta["column_order"] == dd.dcols
        tr = greedy(pipe.predict, dd, kmax)
        for k, step in enumerate(tr, 1):
            rows.append(_row(ctx, network, dd, model_id + "_greedy", k, step["selected"],
                             {"est_reduction_select": step["est_reduction"]}))
    res = pd.DataFrame(rows)
    if len(res):
        res["model_id"] = model_id
        res["runtime_s"] = time.time() - t0
    common.atomic_write_csv(res, os.path.join(out, "decisions.csv"))
    common.atomic_write_csv(pd.DataFrame({"column": dd.dcols, "median_select": dd.med,
                                          "eligible": np.isin(np.arange(len(dd.dcols)), dd.eligible)}),
                            os.path.join(out, "medians.csv"))
    return st.finish(info={"n_rows": int(len(res)), "n_eligible": int(len(dd.eligible))})
