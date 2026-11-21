"""Stage 'simulate': seeded scenarios with the original simulator (protocol §2.2, §9.2).

The distributions and the control point are those of scripts/tools.py: add_risk_parameters,
evaluate_at_EV and the per-activity logic of draw_random_durations / beta_pert_sample.  The
only change is that the beta variates come from an explicit numpy Generator
(``random_state=rng``) instead of NumPy's global RNG.  ``sample_durations`` draws whole
columns at once (vectorised); ``sample_durations_loop`` calls tools.draw_random_durations
row by row with the same seeded beta sampler and is used to verify distributional equivalence.
"""
import functools

import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats as stats

from . import common
from . import config as cfg
from . import data


def beta_pert_sample_rng(optimistic, most_probable, pessimistic, elapsed_time=0, complete_percent=0,
                         size=1000, lamb=4, rng=None):
    """tools.beta_pert_sample with an explicit Generator (identical arithmetic otherwise)."""
    a = optimistic
    b = most_probable
    c = pessimistic
    alpha = 1 + lamb * (b - a) / (c - a)
    beta = 1 + lamb * (c - b) / (c - a)
    samples = stats.beta.rvs(alpha, beta, size=size, random_state=rng)
    samples = a + samples * (c - a)
    return samples if complete_percent == 0 else elapsed_time + (1 - complete_percent) * samples


def sample_durations(G, nodes, n, rng):
    """Vectorised equivalent of n calls to tools.draw_random_durations(G, control=True)."""
    lamb = cfg.SIMULATOR["lamb"]
    out = np.zeros((n, len(nodes)))
    for j, node in enumerate(nodes):
        a = G.nodes[node]
        if not a["pert_duration"] > 0:
            continue                                   # dummy start/end: 0
        if a["done%"] >= 1:
            out[:, j] = a["duration@"]                 # completed
        elif a["done%"] > 0:
            out[:, j] = beta_pert_sample_rng(a["Optimistic"], a["MostLikely"], a["Pessimistic"],
                                             elapsed_time=a["duration@"], complete_percent=a["done%"],
                                             size=n, lamb=lamb, rng=rng)
        else:
            out[:, j] = beta_pert_sample_rng(a["Optimistic"], a["MostLikely"], a["Pessimistic"],
                                             size=n, lamb=lamb, rng=rng)
    return out


def sample_durations_loop(ctx, G, n, rng):
    """Row-by-row reference using tools.draw_random_durations with the seeded sampler."""
    sampler = functools.partial(beta_pert_sample_rng, rng=rng)
    nodes = list(G.nodes())
    rows = np.zeros((n, len(nodes)))
    for m in range(n):
        ctx.tools.draw_random_durations(G, control=True, beta_pert_sample=sampler)
        rows[m] = [G.nodes[i]["duration"] for i in nodes]
    return rows


def scenarios_frame(ctx, G, D, net):
    nodes = list(G.nodes())
    df = pd.DataFrame(D, columns=["duration%s" % i for i in nodes])
    _, ft = ctx.tools.compute_times(G, time="MeanDuration")
    df["baseline_duration"] = np.max(list(ft.values()))
    df["actual_duration"] = net.makespan(D)
    df.insert(0, "scenario_id", np.arange(len(df)))
    return df


def run(ctx, network):
    """Generate scenarios for one network; returns stamp record or None if skipped."""
    G, support = data.control_point(ctx, network)
    nodes = list(G.nodes())
    dcols = ["duration%s" % i for i in nodes]
    from .network import Net
    net = Net(G, dcols)
    original = ctx.is_original(network)
    inputs = {"sm_sha256": common.sha256_file(ctx.sm_path(network)), "original": original}
    out = ctx.stage_dir(network, "simulate")
    st = common.Stamp(out, inputs)
    if not st.begin(ctx.force):
        return None
    G_mean = G.copy()  # evaluate_at_EV set duration to MeanDuration; sampling resets it
    info = {"network_id": network, "seeds": {}, "files": {}, "original_scenarios": original}
    kinds = [("decision_select", cfg.DECISION["n_select"]),
             ("decision_eval", cfg.DECISION["n_eval"]),
             ("sobol_sample", cfg.EXPLAIN["n_sobol"])]
    if not original:
        kinds.insert(0, ("simulate", cfg.SIMULATOR["n_scenarios"]))
    for stage, n in kinds:
        seed = common.derive_seed(network, stage, 0)
        rng = np.random.default_rng(seed)
        D = sample_durations(G_mean, nodes, n, rng)
        df = scenarios_frame(ctx, G_mean, D, net)
        name = "train_test" if stage == "simulate" else stage
        path = common.write_table(df, data.simulated_base(ctx, network, name))
        info["seeds"][name] = seed
        info["files"][name] = {"path": path, "n": int(n), "sha256": common.sha256_file(path)}
    if original:
        # consistency check: the deterministic control point reproduces the constant columns
        df0, dc0, _ = data.load_scenarios(ctx, network)
        sup = support.set_index("column").loc[dc0]
        const = (df0[dc0].max() - df0[dc0].min()).to_numpy() == 0
        diff = np.abs(df0[dc0].to_numpy()[0, const] - sup["support_lo"].to_numpy()[const])
        lo_ok = (df0[dc0].min().to_numpy() >= sup["support_lo"].to_numpy() - 1e-9).all()
        hi_ok = (df0[dc0].max().to_numpy() <= sup["support_hi"].to_numpy() + 1e-9).all()
        info["original_constant_columns_max_abs_diff"] = float(diff.max()) if diff.size else 0.0
        info["original_within_support"] = bool(lo_ok and hi_ok)
    common.atomic_write_csv(support, out + "/support.csv")
    common.atomic_write_json(out + "/simulate_meta.json", info)
    return st.finish(info={"seeds": info["seeds"]})
