#!/usr/bin/python
# Script for Alatmira
# command: python rf1.py >> random_forest_log.txt

import sys
import os
import itertools
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn import linear_model
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


if len(sys.argv) != 2:
    sys.exit("the name of the model is required as an argument, e.g., <<python model_selection.py rfr>>")
            
# Hyperparameters of Random Forest
hyperparam_rf={"n_estimators": [100,300,500],
            "max_depth": [None,5,10]}

# Hyperparameters of Gradient Boosting
hyperparam_gb={"n_estimators": [100,300,500],
            "learning_rate": [0.01,0.1,0.2]}

# Hiperparameters of XGBoost
hyperparam_xgb={"n_estimators": [100,300,500],
            "learning_rate": [0.01,0.1,0.2]}

# Hiperparameters of Explainable Boosting
hyperparam_ebm={"interactions": [0,0.7,0.9],
            "learning_rate": [0.01,0.04,0.1]}

# Hiperparameters of Support Vector Regressor (kernel=rfb)
hyperparam_svr={"C": [0.1,10,100],
                "gamma": ['scale',0.1,1]}

# Hiperparameters of Elastic Net
hyperparam_en={"alpha": [0.01,0.1,1],
            "l1_ratio": [0.1, 0.5, 0.9]}

# Hyperparameters of Bayesian Ridge
hyperparam_br={"alpha_1": [1e-6, 1e-5, 1e-4],
                "lambda_1": [1e-6, 1e-5, 1e-4]}

# Seed for reproducibility
seed=112358

# Name of the model
mdl_name=sys.argv[1]

if mdl_name == 'rf':
    hyperparam=hyperparam_rf
    mdl = RandomForestRegressor
elif mdl_name == 'gb':
    hyperparam=hyperparam_gb
    mdl = GradientBoostingRegressor
elif mdl_name == 'xgb':
    hyperparam=hyperparam_xgb
    mdl = XGBRegressor
elif mdl_name == 'ebm':
    hyperparam=hyperparam_ebm
    mdl = ExplainableBoostingRegressor
elif mdl_name == 'sv':
    hyperparam=hyperparam_svr
    mdl = SVR
elif mdl_name == 'en':
    hyperparam=hyperparam_en
    mdl = ElasticNet
elif mdl_name == 'br':
    hyperparam=hyperparam_br
    mdl = linear_model.BayesianRidge
else:
    sys.exit(f"Model {mdl_name} is not recognized.")

# Projects dataset for model selection
# Check if the dataset file exists
dataset_path = os.path.join('./results', 'model_selection_projects.csv')
if not os.path.exists(dataset_path):
    sys.exit(f"Dataset file {dataset_path} does not exist.")
else:
    df_projects=pd.read_csv(os.path.join('./results','model_selection_projects.csv'))

df_selection=df_projects[df_projects['ModelSelection']] # Filter ModelSelection True

# Set the hyperparameters to explore
param_keys=list(hyperparam.keys())
param_values=list(hyperparam.values())

for sp in df_selection['SP'].unique():

    # List of projects for training and testing
    for project in df_selection.loc[df_selection['SP'] == sp,'Project']:
        dataset_path = os.path.join('./results', f"{project}_simulation_e1.csv")
        if not os.path.exists(dataset_path):
            sys.exit(f"Simulation data file {dataset_path} does not exist.")
        else:
            print(project)
            df=pd.read_csv(dataset_path)