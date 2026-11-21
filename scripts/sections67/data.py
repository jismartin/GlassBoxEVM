"""Paths, network catalogue, control point, scenario loading and train/test split."""
import copy
import importlib
import os
import sys

import numpy as np
import pandas as pd

from . import common
from . import config as cfg
from .network import Net


def default_repo():
    # scripts/sections67/data.py -> repository root
    return os.path.dirname(os.path.dirname(common.PKG_DIR))


def default_results(repo):
    env = os.environ.get("SECTIONS67_RESULTS")
    if env:
        return env
    return os.path.join(os.path.dirname(os.path.abspath(repo)), "GlassBoxEVM_sections67_results")


_TOOLS = {}


def get_tools(repo):
    """Import the original simulator module scripts/tools.py (never modified)."""
    key = os.path.abspath(repo)
    if key not in _TOOLS:
        path = os.path.join(repo, "scripts")
        if path not in sys.path:
            sys.path.insert(0, path)
        _TOOLS[key] = importlib.import_module("tools")
    return _TOOLS[key]


class Ctx:
    def __init__(self, repo, results, cohort, threads=1, force=False):
        self.repo = os.path.abspath(repo)
        self.results = os.path.abspath(results)
        self.cohort = cohort
        self.threads = int(threads)
        self.force = bool(force)
        rp = os.path.realpath(self.results)
        src = os.path.realpath(os.path.join(self.repo, "scripts"))
        if rp == src or rp.startswith(src + os.sep):
            raise ValueError("results directory must lie outside the repository source tree")
        self.tools = get_tools(self.repo)
        self._catalogue = None
        self._evaluation = None
        self._test = None

    # -- paths
    @property
    def cohort_dir(self):
        return os.path.join(self.results, self.cohort)

    def net_dir(self, network):
        return os.path.join(self.cohort_dir, "networks", network)

    def stage_dir(self, network, stage, model=None):
        d = os.path.join(self.net_dir(network), stage)
        return os.path.join(d, model) if model else d

    # -- catalogues
    @property
    def catalogue(self):
        if self._catalogue is None:
            df = pd.read_csv(os.path.join(self.repo, cfg.COHORTS["catalogue"]))
            df["network_id"] = df["FileName"].str.replace(".sm", "", regex=False)
            self._catalogue = df
        return self._catalogue

    @property
    def evaluation(self):
        if self._evaluation is None:
            df = pd.read_csv(os.path.join(self.repo, cfg.COHORTS["evaluation_list"]), index_col=0)
            df["network_id"] = df["FileName"].str.replace(".sm", "", regex=False)
            self._evaluation = df
        return self._evaluation

    def subset_of(self, network):
        row = self.catalogue.loc[self.catalogue["network_id"] == network]
        if len(row) != 1:
            raise ValueError("network %s not found exactly once in catalogue" % network)
        return str(row.iloc[0]["SubSet"])

    def is_original(self, network):
        """True for the 400 originally simulated networks (scenarios in montecarlo/)."""
        row = self.catalogue.loc[self.catalogue["network_id"] == network]
        return bool(len(row) == 1 and row.iloc[0]["sample"]) and os.path.exists(self.original_path(network))

    def original_path(self, network):
        return os.path.join(self.repo, cfg.SIMULATOR["original_scenarios"].format(network=network))

    def sm_path(self, network):
        p = os.path.join(self.repo, "PSPLIB_sample", network + ".sm")
        if os.path.exists(p):
            return p
        return os.path.join(self.repo, "PSPLIB_complete", network + ".sm")

    @property
    def test_runs(self):
        if self._test is None:
            self._test = pd.read_csv(os.path.join(self.repo, cfg.SIMULATOR["test_runs_file"]))["test_runs"].values
        return self._test


# ------------------------------------------------------------------ graph and control point
def load_graph(ctx, network):
    return ctx.tools.read_sm_file(ctx.sm_path(network))


def control_point(ctx, network):
    """Apply tools.add_risk_parameters and tools.evaluate_at_EV; return (G, support table).

    The support table gives, per node, status (dummy / completed / in_progress / not_started),
    the beta-PERT parameters and the theoretical support of the simulated duration column,
    including the affine transformation of activities in progress.
    """
    s = cfg.SIMULATOR
    G = load_graph(ctx, network)
    ctx.tools.add_risk_parameters(G, optim=s["optim"], mostprob=s["mostprob"], pessim=s["pessim"])
    ctx.tools.evaluate_at_EV(G, EV=s["EV"], Ti=s["Ti"], N=s["N_ev"])
    rows = []
    for n in G.nodes():
        a = G.nodes[n]
        O, M, P = a["Optimistic"], a["MostLikely"], a["Pessimistic"]
        done, at = a["done%"], a["duration@"]
        if not a["pert_duration"] > 0:
            status, lo, hi = "dummy", 0.0, 0.0
        elif done >= 1:
            status, lo, hi = "completed", at, at
        elif done > 0:
            status, lo, hi = "in_progress", at + (1 - done) * O, at + (1 - done) * P
        else:
            status, lo, hi = "not_started", O, P
        rows.append({"node": n, "column": "duration%s" % n, "status": status,
                     "pert_duration": a["pert_duration"], "optimistic": O, "most_likely": M,
                     "pessimistic": P, "mean_duration": a["MeanDuration"], "done_frac": done,
                     "duration_at": at, "support_lo": lo, "support_hi": hi})
    return G, pd.DataFrame(rows)


# ------------------------------------------------------------------ scenarios
def simulated_base(ctx, network, kind):
    return os.path.join(ctx.stage_dir(network, "simulate"), kind)


def load_scenarios(ctx, network):
    """Training/test scenarios: original montecarlo file, or the 'simulate' stage output."""
    if ctx.is_original(network):
        path = ctx.original_path(network)
        df = pd.read_csv(path, index_col=0)
    else:
        base = simulated_base(ctx, network, "train_test")
        df = common.read_table(base).set_index("scenario_id")
        path = base
    df.index.name = None
    dcols = [c for c in df.columns if c.startswith("duration")]
    return df, dcols, path


def scenario_input_hash(ctx, network):
    if ctx.is_original(network):
        return common.sha256_file(ctx.original_path(network))
    base = simulated_base(ctx, network, "train_test")
    for e in (".parquet", ".csv.gz"):
        if os.path.exists(base + e):
            return common.sha256_file(base + e)
    raise FileNotFoundError(base)


def load_extra_sample(ctx, network, kind):
    """decision_select / decision_eval / sobol_sample matrices from the simulate stage."""
    df = common.read_table(simulated_base(ctx, network, kind))
    dcols = [c for c in df.columns if c.startswith("duration")]
    return df, dcols


def split(df, test_runs):
    """Exactly as model_evaluation.py."""
    df_train = df.loc[df.index.difference(test_runs)]
    df_test = df.loc[test_runs]
    return df_train, df_test


def constant_mask(df_train, dcols):
    """Rule of the original fastai pipeline: nunique <= 1 in training."""
    return np.array([df_train[c].nunique() <= 1 for c in dcols])


def stage_indices(n_avail_idx, network, stage, n):
    """Draw n indices without replacement from n_avail_idx with the stage generator."""
    pool = np.asarray(n_avail_idx)
    rng = common.stage_rng(network, stage, 0)
    m = min(n, len(pool))
    return np.sort(rng.choice(pool, size=m, replace=False)) if m < len(pool) else np.sort(pool.copy())


def network_bundle(ctx, network):
    """Everything most stages need, computed deterministically from inputs."""
    df, dcols, path = load_scenarios(ctx, network)
    G, support = control_point(ctx, network)
    support = support.set_index("column").loc[dcols].reset_index()
    net = Net(G, dcols)
    df_train, df_test = split(df, ctx.test_runs)
    const = constant_mask(df_train, dcols)
    return {"df": df, "dcols": dcols, "path": path, "G": G, "support": support, "net": net,
            "df_train": df_train, "df_test": df_test, "const": const,
            "subset": ctx.subset_of(network)}
