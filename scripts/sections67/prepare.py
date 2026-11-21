"""Stage 'prepare': network manifests, confirmatory sampling with substitutes (§2.1, §9.2)."""
import os

import numpy as np
import pandas as pd

from . import common
from . import config as cfg
from . import data

STAGE_SEEDS = ["simulate", "local_test", "pd_background", "sobol_sample", "decision_select", "decision_eval"]


def _seed_text(network):
    return ";".join("%s=%d" % (s, common.derive_seed(network, s, 0)) for s in STAGE_SEEDS)


def check_network(ctx, network):
    """Allowed pre-exclusions only: unreadable file, control point, topological indicators."""
    try:
        G = data.load_graph(ctx, network)
    except Exception as e:  # noqa: BLE001
        return "unreadable_file: %s" % e
    try:
        data.control_point(ctx, network)
    except Exception as e:  # noqa: BLE001
        return "control_point_failed: %s" % e
    try:
        ctx.tools.compute_topological_indicators(G)
    except Exception as e:  # noqa: BLE001
        return "topology_failed: %s" % e
    return ""


def candidate_permutation(ctx):
    cat = ctx.catalogue
    rows = []
    for subset, k in cfg.SEEDS["subset_k"].items():
        cand = cat[(cat["SubSet"] == subset) & (~cat["sample"].astype(bool))]["network_id"].tolist()
        rng = np.random.default_rng([cfg.SEEDS["network_sampling_seed"], k])
        perm = rng.permutation(len(cand))
        for rank, p in enumerate(perm):
            rows.append({"subset": subset, "rank": rank, "network_id": cand[p], "n_candidates": len(cand)})
    return pd.DataFrame(rows)


def run(ctx, network_list=None):
    out = ctx.cohort_dir
    common.ensure_dir(out)
    rows = []
    if ctx.cohort == "confirmatory":
        perm = candidate_permutation(ctx)
        perm["selected"] = False
        perm["substitute_order"] = np.nan
        n_target = cfg.COHORTS["n_confirmatory_per_subset"]
        for subset in cfg.COHORTS["subsets"]:
            sub = perm[perm["subset"] == subset].sort_values("rank")
            n_ok = 0
            n_sub = 0
            for i, r in sub.iterrows():
                if n_ok >= n_target:
                    n_sub += 1
                    perm.loc[i, "substitute_order"] = n_sub
                    continue
                reason = check_network(ctx, r["network_id"])
                status = "excluded" if reason else "included"
                if not reason:
                    n_ok += 1
                    perm.loc[i, "selected"] = True
                rows.append({"cohort": ctx.cohort, "subset": subset, "network_id": r["network_id"],
                             "rank": int(r["rank"]), "status": status, "exclusion_reason": reason})
        perm["numpy_version"] = np.__version__
        perm["algorithm"] = "numpy.random.default_rng([%d, k]).permutation" % cfg.SEEDS["network_sampling_seed"]
        common.atomic_write_csv(perm, os.path.join(out, "candidate_permutation.csv"))
    else:
        if ctx.cohort == "exploratory" and network_list is None:
            ids = ctx.evaluation["network_id"].tolist()
        elif network_list is not None:
            ids = pd.read_csv(network_list)["network_id"].astype(str).tolist()
        else:
            raise ValueError("cohort %s needs --network-list (CSV with a network_id column)" % ctx.cohort)
        for nid in ids:
            reason = check_network(ctx, nid)
            rows.append({"cohort": ctx.cohort, "subset": ctx.subset_of(nid), "network_id": nid, "rank": np.nan,
                         "status": "excluded" if reason else "included", "exclusion_reason": reason})
    man = pd.DataFrame(rows)
    man["source_sha256"] = [common.sha256_file(ctx.sm_path(n)) if os.path.exists(ctx.sm_path(n)) else ""
                            for n in man["network_id"]]
    man["scenario_source"] = ["original_montecarlo" if ctx.is_original(n) else "simulate_stage"
                              for n in man["network_id"]]
    man["seeds"] = [_seed_text(n) for n in man["network_id"]]
    man["config_hash"] = cfg.config_hash()
    common.atomic_write_csv(man, os.path.join(out, "network_manifest.csv"))
    common.atomic_write_text(os.path.join(out, "config_snapshot.json"), cfg.config_json())
    return man


def included_networks(ctx):
    path = os.path.join(ctx.cohort_dir, "network_manifest.csv")
    man = pd.read_csv(path)
    return man[man["status"] == "included"]["network_id"].tolist()
