#!/usr/bin/python
# Script for Altamira
# Evaluate one fixed hyperparameterization of a model on one evaluation project.
#
# We run FastAI locally because the RelMSE is quite different on the cluster than locally,
# even reviewing all the parameterization to guarantee replicability.
# A possible cause is that the data are different or corrupted on that machine (wrong file, different split, or row misalignment).


import argparse
import os
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
df_test_runs = pd.read_csv(f"{args.path_montecarlo}/test_runs.csv")

# Reset seeds for replicability before each project evaluation
reset_seeds(seed)

# Read the simulation data for the project
df = pd.read_csv(f"{args.path_montecarlo}/{project_name}_simulation_01.csv", index_col=0)

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
        hyperparameters["n_jobs"] = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
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
