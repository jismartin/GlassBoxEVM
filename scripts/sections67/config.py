"""Executable configuration of the Sections 6-7 pipeline (protocol v1.0, §9.1).

This module replaces ``configs/sections_6_7.yaml`` proposed in the protocol: YAML/JSON
files cannot be placed into the repository with the tools available, so the full
configuration lives here as plain Python literals.  ``config_hash()`` is a SHA-256 of the
canonical JSON serialisation of ``CONFIG`` and is written into every result stamp.

Retained configurations are copied from ``model_evaluation/model_evaluation.py`` and from
``model_evaluation/hyperparameters/lhs_<model>_parameters.csv`` (read with pandas defaults,
exactly as model_evaluation.py does).  ``models.load_hyperparameters`` re-reads the CSV at
run time and refuses to fit if the values differ from the ones frozen here.
"""
import hashlib
import json

PROTOCOL_VERSION = "1.0"
PROTOCOL_PATH = "protocols/sections_6_7.md"

# ---------------------------------------------------------------- seeds (§9.2)
SEEDS = {
    "training_seed": 112358,            # model_evaluation.py, reset before each fit
    "network_sampling_seed": 20261119,  # numpy.random.default_rng([20261119, k])
    "subset_k": {"J30": 0, "J60": 1, "J90": 2, "J120": 3},
    "analysis_root_seed": 20261121,     # SHA-256('root|network_id|stage|replicate')
    "stages": ["simulate", "local_test", "pd_background", "sobol_sample", "sobol_perm",
               "decision_select", "decision_eval", "independent_validation", "stability",
               "network_bootstrap"],
}

# ---------------------------------------------------------------- simulator (§2.2)
SIMULATOR = {
    "optim": 0.8, "mostprob": 1.1, "pessim": 1.2, "lamb": 4,     # tools.add_risk_parameters
    "EV": 0.15, "Ti": 0, "N_ev": 10000,                          # tools.evaluate_at_EV
    "n_scenarios": 10000, "test_runs_file": "montecarlo/test_runs.csv",
    "original_scenarios": "montecarlo/{network}_simulation_01.csv",
    "historical_simulation_seed": None,  # does not exist (protocol §2.2)
}

# ---------------------------------------------------------------- cohorts (§2.1)
COHORTS = {
    "n_confirmatory_per_subset": 80,
    "subsets": ["J30", "J60", "J90", "J120"],
    "catalogue": "PSPLIB_complete/psplib_complete.csv",
    "evaluation_list": "model_evaluation/project_evaluation.csv",
    "allowed_exclusions": ["unreadable_file", "control_point_failed", "topology_failed"],
}

# ---------------------------------------------------------------- models (§3.2)
HYPERPARAMETER_DIR = "model_evaluation/hyperparameters"
MODELS = {
    "EN-36": {
        "family": "en", "lhs": 36,
        "hyperparameters": {"alpha": 0.0001318256738556, "l1_ratio": 0.91},
        "fixed": {"random_state": 112358, "max_iter": 50000},
        "features": "PolynomialFeatures(degree=2, interaction_only=True, include_bias=False) on all "
                    "duration columns (uncentred), then StandardScaler",
    },
    "EBM-24": {
        "family": "ebm", "lhs": 24,
        "hyperparameters": {"interactions": 0.8729999999999999, "learning_rate": 0.0156894818692829,
                            "max_leaves": 3, "min_samples_leaf": 3, "early_stopping_rounds": 74},
        "fixed": {"random_state": 112358, "n_jobs": "SLURM_CPUS_PER_TASK or 1"},
        "features": "StandardScaler on all duration columns; ExplainableBoostingRegressor defaults "
                    "(interpret 0.7.2) for every option not listed",
    },
    "fastai-41": {
        "family": "fastai", "lhs": 41,
        "hyperparameters": {"lr_max": 0.0015848931924611, "layer1": 120, "layer2": 128,
                            "wd": 0.0007943282347242},
        "fixed": {"seed": 112358, "set_seed_reproducible": True, "layers": "int((layer//5)*5)",
                  "valid_pct": 0.1, "bs": 64, "num_workers": 0, "epochs": 100,
                  "fit": "fit_one_cycle", "early_stopping": "valid_loss patience 10",
                  "procs": ["Categorify", "FillMissing", "Normalize"], "loss": "MSELossFlat",
                  "config": "tabular_config()", "drop_constant_columns": True,  # 'fix A'
                  "torch_threads": "SLURM_CPUS_PER_TASK or 1"},
        "features": "constant training columns dropped (nunique <= 1), fastai Normalize",
    },
    "OLS": {
        "family": "ols", "lhs": None, "hyperparameters": {},
        "fixed": {"fit_intercept": True},
        "features": "StandardScaler on variable (non-constant in training) duration columns, "
                    "LinearRegression",
    },
    "EBM-24-noint": {
        "family": "ebm", "lhs": 24, "ablation_of": "EBM-24",
        "hyperparameters": {"interactions": 0.8729999999999999, "learning_rate": 0.0156894818692829,
                            "max_leaves": 3, "min_samples_leaf": 3, "early_stopping_rounds": 74},
        "override": {"interactions": 0},
        "fixed": {"random_state": 112358, "n_jobs": "SLURM_CPUS_PER_TASK or 1"},
        "features": "as EBM-24 with interactions=0",
    },
    "EN-36-main": {
        "family": "en", "lhs": 36, "ablation_of": "EN-36",
        "hyperparameters": {"alpha": 0.0001318256738556, "l1_ratio": 0.91},
        "fixed": {"random_state": 112358, "max_iter": 50000},
        "features": "no product terms: StandardScaler on all duration columns, ElasticNet",
    },
    "EN-36-centred": {
        "family": "en", "lhs": 36, "ablation_of": "EN-36",
        "hyperparameters": {"alpha": 0.0001318256738556, "l1_ratio": 0.91},
        "fixed": {"random_state": 112358, "max_iter": 50000},
        "features": "subtract training means, then pairwise products, then StandardScaler",
    },
}
FOCAL_MODELS = ["EN-36", "EBM-24", "fastai-41"]
ALL_MODELS = ["EN-36", "EBM-24", "fastai-41", "OLS", "EBM-24-noint", "EN-36-main", "EN-36-centred"]

# ---------------------------------------------------------------- structure (§4)
STRUCTURE = {
    "critical_tol": 1e-9, "sensitivity_tols": [1e-7, 1e-5],
    "intermediate_ci": [0.05, 0.95],
    "decomposition": {"enabled": True, "n_grid": 21, "q_lo": 0.01, "q_hi": 0.99,
                      "background": "all_scenarios", "min_L_for_ratio": 1e-3},
}

# ---------------------------------------------------------------- explanations (§6.1, §6.2)
EXPLAIN = {
    "n_local": 500, "steps_sd": [-1.0, -0.5, 0.5, 1.0], "sd_ddof": 0,
    "min_effective_fraction": 0.1, "violation_tol": 0.05, "null_change_rel_tol": 1e-9,
    "n_pd_background": 500, "pd_n_grid": 21, "pd_q_lo": 0.01, "pd_q_hi": 0.99,
    "pd_extrapolation": "linear (interp_lin of the exploratory code)",
    "n_sobol": 10000, "n_s1_background": 500, "n_perm_seeds": 3,
    "predict_chunk_rows": 4000,
}

# ---------------------------------------------------------------- resolution checks (§7.1)
RESOLUTION = {
    "network_rule": ("per PSPLIB set, the pilot network with C > 0 nearest the 50% quantile of C; "
                     "plus the maximum-C illustrative network"),
    "grids": [21, 41], "s1_backgrounds": [500, 2000],
    "support_limits_grid": True, "models": ["EN-36", "EBM-24", "fastai-41"],
    "independent_sample": "sobol_sample",
}

# ---------------------------------------------------------------- decision (§6.3)
DECISION = {
    "n_select": 5000, "n_eval": 5000, "k_values": [1, 2, 3],
    "policies_static": ["CI", "SSI"], "exhaustive_subsets": ["J30", "J60"],
    "p90_method": "linear", "h4_k": 3,
}

# ---------------------------------------------------------------- inference (§5, §9.5)
INFERENCE = {
    "n_boot": 10000, "ci": [2.5, 97.5], "family_alpha": 0.05,
    "holm_family": ["H2", "H3", "H4"], "primary": "H1",
    "h2_contrast": ["J30", "J120"], "cooks_threshold": "4/n",
    "theilsen_method": "separate",
}

RESOURCES = {"threads_per_task": "SLURM_CPUS_PER_TASK or 1 (BLAS/OpenMP limited to it)"}

CONFIG = {
    "protocol_version": PROTOCOL_VERSION, "protocol_path": PROTOCOL_PATH,
    "seeds": SEEDS, "simulator": SIMULATOR, "cohorts": COHORTS,
    "hyperparameter_dir": HYPERPARAMETER_DIR, "models": MODELS,
    "focal_models": FOCAL_MODELS, "all_models": ALL_MODELS,
    "structure": STRUCTURE, "explain": EXPLAIN, "resolution": RESOLUTION, "decision": DECISION,
    "inference": INFERENCE, "resources": RESOURCES,
}


def config_json():
    """Canonical JSON text of CONFIG (sorted keys, repr-exact floats)."""
    return json.dumps(CONFIG, sort_keys=True, separators=(",", ":"))


def config_hash():
    return hashlib.sha256(config_json().encode("utf-8")).hexdigest()
