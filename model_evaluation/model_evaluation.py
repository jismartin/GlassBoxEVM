#!/usr/bin/python
# Script for Altamira
# Evaluate one fixed hyperparameterization of a model on one evaluation project.
#
# All models, FastAI included, are run on Altamira through single-core SLURM array
# jobs (model_evaluation/jobs/). Each run prints its execution environment and the
# SHA-256 of the input files to stdout (SLURM .out log).


import argparse
import hashlib
import os
import platform as py_platform
import socket as py_socket
import sys as py_sys
import random
import time

import numpy as np
import pandas as pd
import torch
from fastai.metrics import mse
from fastai.tabular.all import *
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor


# ---------------- CLI ----------------
parser = argparse.ArgumentParser(
    description="Evaluate one model with one selected hyperparameterization on one project."
)
parser.add_argument(
    "model_name",
    type=str,
    choices=["rf", "xgb", "ebm", "sv", "en", "fastai"],
    help="Short name of the model: rf, xgb, ebm, sv, en, fastai",
)
parser.add_argument(
    "hyperparameter_id",
    type=str,
    help="Selected hyperparameterization from lhs_<model_name>_parameters.csv",
)
parser.add_argument(
    "project_name",
    type=str,
    help="Project name or file name, for example j301_9 or j301_9.sm",
)
parser.add_argument(
    "--path-results",
    type=str,
    default="./model_evaluation/results",
    help="Directory where the output CSV will be written.",
)
parser.add_argument(
    "--path-hyperparameters",
    type=str,
    default="./model_evaluation/hyperparameters",
    help="Directory containing lhs_<model_name>_parameters.csv.",
)
parser.add_argument(
    "--path-montecarlo",
    type=str,
    default="./montecarlo",
    help="Directory containing project simulation CSVs and test_runs.csv.",
)
parser.add_argument(
    "--path-project-evaluation",
    type=str,
    default="./model_evaluation/project_evaluation.csv",
    help="CSV containing the project list and subset assignment.",
)
args = parser.parse_args()

mdl_name = args.model_name
hyperparameter_id = int(args.hyperparameter_id)
project_name = args.project_name.removesuffix(".sm")
project_file_name = f"{project_name}.sm"

# ---------------- Setup ----------------
seed = 112358


def reset_seeds(seed_value: int) -> None:
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def print_environment() -> None:
    import fastai
    import interpret
    import sklearn
    import xgboost

    print("---- Execution environment ----")
    print(f"hostname: {py_socket.gethostname()}")
    print(f"platform: {py_platform.platform()}")
    print(f"python: {py_sys.version.split()[0]}")
    for lib in (np, pd, torch, fastai, sklearn, xgboost, interpret):
        print(f"{lib.__name__}: {getattr(lib, '__version__', 'unknown')}")
    print(f"SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID', '-')}")
    print(f"SLURM_ARRAY_TASK_ID: {os.environ.get('SLURM_ARRAY_TASK_ID', '-')}")
    print(f"SLURM_CPUS_PER_TASK: {os.environ.get('SLURM_CPUS_PER_TASK', '-')}")
    print(f"torch threads: {torch.get_num_threads()}")
    print("-------------------------------")


n_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
if mdl_name == "fastai":
    torch.set_num_threads(n_threads)
print_environment()

os.makedirs(args.path_results, exist_ok=True)

# Load the model
if mdl_name == "rf":
    mdl = RandomForestRegressor
elif mdl_name == "xgb":
    mdl = XGBRegressor
elif mdl_name == "ebm":
    mdl = ExplainableBoostingRegressor
elif mdl_name == "sv":
    mdl = SVR
elif mdl_name == "en":
    mdl = ElasticNet
elif mdl_name == "fastai":
    mdl = None
else:
    raise ValueError(f"Model {mdl_name} not recognized.")

# ---------------- Data ----------------
project_evaluation_path = args.path_project_evaluation
if not os.path.exists(project_evaluation_path):
    typo_path = project_evaluation_path.replace("project_evaluation.csv", "proyect_evaluation.csv")
    if os.path.exists(typo_path):
        project_evaluation_path = typo_path
    else:
        raise FileNotFoundError(project_evaluation_path)

df_projects = pd.read_csv(project_evaluation_path)
project_row = df_projects.loc[
    df_projects["FileName"].astype(str).str.lower() == project_file_name.lower()
]
if project_row.empty:
    raise ValueError(f"Project {project_file_name} not found in {project_evaluation_path}.")
if len(project_row) > 1:
    raise ValueError(f"Project {project_file_name} appears more than once in {project_evaluation_path}.")

subset = project_row.iloc[0]["SubSet"]

# Read the LHS parameters
mdl_lhs_params = pd.read_csv(
    f"{args.path_hyperparameters}/lhs_{mdl_name}_parameters.csv",
    index_col=0,
)

# Read test runs
test_runs_path = f"{args.path_montecarlo}/test_runs.csv"
df_test_runs = pd.read_csv(test_runs_path)

# Reset seeds for replicability before each project evaluation
reset_seeds(seed)

# Read the simulation data for the project
simulation_path = f"{args.path_montecarlo}/{project_name}_simulation_01.csv"
df = pd.read_csv(simulation_path, index_col=0)
print(f"sha256 {os.path.basename(simulation_path)}: {sha256_file(simulation_path)}")
print(f"sha256 {os.path.basename(test_runs_path)}: {sha256_file(test_runs_path)}")

# Training data
df_train = df.loc[df.index.difference(df_test_runs["test_runs"].values)]
Xtrain = df_train[[cl for cl in df_train.columns if cl.startswith("duration")]]
Ytrain = df_train["actual_duration"]

# Testing data
df_test = df.loc[df_test_runs["test_runs"].values]
Xtest = df_test[[cl for cl in df_test.columns if cl.startswith("duration")]]
Ytest = df_test["actual_duration"]

print(f"Running {mdl_name} for {project_name} with {hyperparameter_id}-LHShyperparameters")

# For computing the run time
start_time = time.time()

# Load the model's hyperparameters
hyperparameters = mdl_lhs_params.iloc[[hyperparameter_id]].to_dict(orient="records")[0]

if mdl_name == "fastai":
    set_seed(seed, reproducible=True)

    # Drop zero-variance inputs (activities completed at the control point): they are constant,
    # carry no information, and fastai's Normalize maps them to large non-zero test values
    # (statistics on float32 training data vs float64 test data; see test/altamira_fastai_lhs10).
    const_cols = [c for c in df_train.columns if c.startswith("duration") and df_train[c].nunique() <= 1]
    print(f"fastai: dropping {len(const_cols)} constant input columns")
    df_train = df_train.drop(columns=const_cols)
    df_test = df_test.drop(columns=const_cols)
    splits = RandomSplitter(valid_pct=0.1, seed=seed)(range_of(df_train))
    to = TabularPandas(
        df_train,
        procs=[Categorify, FillMissing, Normalize],
        cat_names=[],
        cont_names=[col for col in df_train.columns if col.startswith("duration")],
        y_names="actual_duration",
        splits=splits,
    )
    dls = to.dataloaders(bs=64, num_workers=0)

    layers = [
        int((hyperparameters["layer1"] // 5) * 5),
        int((hyperparameters["layer2"] // 5) * 5),
    ]
    wd = float(hyperparameters["wd"])
    lr_max = float(hyperparameters["lr_max"])

    learn = tabular_learner(
        dls,
        layers=layers,
        metrics=mse,
        config=tabular_config(),
        wd=wd,
        loss_func=MSELossFlat(),
        cbs=[EarlyStoppingCallback(monitor="valid_loss", patience=10)],
    )

    learn.fit_one_cycle(100, lr_max=lr_max)

    test_dl = dls.test_dl(df_test, with_labels=True, shuffle=False)
    test_preds, targs = learn.get_preds(dl=test_dl)
    mse_model = mse(test_preds, targs).item()

    Ybase = np.repeat(df_train["actual_duration"].mean(), df_test.shape[0])
    mse_base = mean_squared_error(df_test["actual_duration"].values, Ybase)
    rel_mse = mse_model / mse_base if mse_base != 0 else np.inf

else:
    if mdl_name in ["rf", "xgb", "ebm", "en"]:
        hyperparameters["random_state"] = seed
    if mdl_name in ["rf", "xgb", "ebm"]:
        hyperparameters["n_jobs"] = n_threads
    if mdl_name == "en":
        hyperparameters["max_iter"] = 50000

    if mdl_name == "en":
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        Xtrain_poly = poly.fit_transform(Xtrain)
        Xtest_poly = poly.transform(Xtest)

        scaler = StandardScaler()
        Xtrain_scaled = scaler.fit_transform(Xtrain_poly)
        Xtest_scaled = scaler.transform(Xtest_poly)
    else:
        scaler = StandardScaler()
        Xtrain_scaled = scaler.fit_transform(Xtrain)
        Xtest_scaled = scaler.transform(Xtest)

    model = mdl(**hyperparameters)
    model.fit(Xtrain_scaled, Ytrain)
    Ypred = model.predict(Xtest_scaled)

    dummy = DummyRegressor(strategy="mean")
    dummy.fit(Xtrain_scaled, Ytrain)
    Ybase = dummy.predict(Xtest_scaled)
    mse_base = mean_squared_error(Ytest, Ybase)
    mse_model = mean_squared_error(Ytest, Ypred)
    rel_mse = mse_model / mse_base if mse_base != 0 else np.inf

elapsed_time = round(time.time() - start_time, 0)

# ---------------- Save results ----------------
output_file = (
    f"{args.path_results}/model_evaluation_{mdl_name}_lhs_{args.hyperparameter_id}"
    f"_subset_{subset}_project_{project_name}.csv"
)
pd.DataFrame(
    [
        [project_name, subset, mdl_name, hyperparameter_id]
        + [hyperparameters[c] for c in mdl_lhs_params.columns]
        + [rel_mse, elapsed_time]
    ],
    columns=["project", "subset", "model", "lhs"]
    + mdl_lhs_params.columns.tolist()
    + ["RelMSE", "time"],
).to_csv(output_file, index=False)
