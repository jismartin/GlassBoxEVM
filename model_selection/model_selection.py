#!/usr/bin/python
# Script for Altamira
# We run FastAI locally because the RelMSE is quite different on the cluster than locally,
# even reviewing all the parameterization to guarantee replicability.
# A possible cause is that the data are different or corrupted on that machine (wrong file, different split, or row misalignment).


import sys, os, time, argparse, random
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor

from interpret.glassbox import ExplainableBoostingRegressor

from xgboost import XGBRegressor

from fastai.tabular.all import *
from fastai.metrics import mse
import torch


# ---------------- CLI ----------------
# Parse command line arguments
parser = argparse.ArgumentParser(description='Model selection script for Altamira')
# Arguments
parser.add_argument('model_name', type=str, choices=['rf', 'xgb', 'ebm', 'sv', 'en', 'fastai'],
                    help='Short name of the model (rf, xgb, ebm, sv, en, fastai)')
parser.add_argument('project_name', type=str, help='Project name for selection model')
args = parser.parse_args()
# Get parsed values
mdl_name = args.model_name
project_name = args.project_name

# ---------------- Setup ----------------
# Seed for replicability
seed = 112358
np.random.seed(seed); random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)

path_results = './model_selection/results'
path_hyperparameters = './model_selection/hyperparameters'
path_montecarlo = './montecarlo'
os.makedirs(path_results, exist_ok=True)

# Load the model
if mdl_name == 'rf':
    mdl = RandomForestRegressor
elif mdl_name == 'xgb':
    mdl = XGBRegressor
elif mdl_name == 'ebm':
    mdl = ExplainableBoostingRegressor
elif mdl_name == 'sv':
    mdl = SVR
elif mdl_name == 'en':
    mdl = ElasticNet
elif mdl_name == 'fastai':
    mdl = None
else:
    raise ValueError(f"Model {mdl_name} not recognized.")

# ---------------- Data ----------------
# Read the simulation data
df = pd.read_csv(f"{path_montecarlo}/{project_name}_simulation_01.csv",index_col=0)

# Read the LHS parameters
mdl_lhs_params = pd.read_csv(f"{path_hyperparameters}/lhs_{mdl_name}_parameters.csv",index_col=0)

# Read test runs
df_test_runs= pd.read_csv(f"{path_montecarlo}/test_runs.csv")

# Training data
df_train=df.loc[df.index.difference(df_test_runs['test_runs'].values)]
Xtrain=df_train[[cl for cl in df_train.columns if cl.startswith('duration')]] # features correspondint to the activities' durations
Ytrain=df_train['actual_duration'] # The objective function: the actual duration of the project

# Testing data
df_test=df.loc[df_test_runs['test_runs'].values]
Xtest=df_test[[cl for cl in df_test.columns if cl.startswith('duration')]] 
Ytest=df_test['actual_duration'] 

mse_test = []

# ---------------- Train and test models ----------------
# Test all combinations of hyperparameters
for lhs_i in mdl_lhs_params.index[:1]:

    print(f"Running {mdl_name} for {project_name} with {lhs_i}-LHShyperparameters")

    # For comuting the run time
    start_time = time.time()

    # Load the model's hyperparameters
    hyperparameters = mdl_lhs_params.iloc[[lhs_i]].to_dict(orient='records')[0] # Using `.loc[i, :]` returns a Series, which may upcast ints to floats — use `.iloc[[i]]` for dtype-safe row extraction.
     
    if mdl_name == 'fastai':
        set_seed(seed, reproducible=True)
    
        # Prepare training/validation set
        splits = RandomSplitter(valid_pct=0.1, seed=seed)(range_of(df_train)) # We assign the validation set to 10% of the training data
        to = TabularPandas(df_train, 
                procs=[Categorify,FillMissing,Normalize],
                cat_names = [],
                cont_names = [ col for col in df_train.columns if col.startswith('duration')],
                y_names='actual_duration',
                splits=splits)
        # Create DataLoaders, 'dls' will have 'train' and 'valid' DataLoaders
        dls = to.dataloaders(bs=64, num_workers=0) # Try other batch sizes?

        # Load layers
        layers=[]
        layers.append(int((hyperparameters['layer1']//5)*5)) # Round to the nearest multiple of 5
        layers.append(int((hyperparameters['layer2']//5)*5)) # Round to the nearest multiple of 5
        wd = float(hyperparameters['wd'])
        lr_max = float(hyperparameters['lr_max'])
        
        # (2) Train the model
        epochs = 100
        learn = tabular_learner(
            dls,
            layers=layers,
            metrics=mse,
            config=tabular_config(), # Dropout for embedding layers
            wd=wd, # Weight decay
            loss_func=MSELossFlat(), # Mean Squared Error as regression loss
            cbs=[EarlyStoppingCallback(monitor='valid_loss', patience=10)]
        )
        
        # Find the best learning rate
        #lr_suggestion = learn.lr_find(show_plot=False)
        # and train the model
        learn.fit_one_cycle(epochs, lr_max=lr_max)

        # (3) Evaluate the model on the test dataset
        test_dl = dls.test_dl(df_test, with_labels=True, shuffle=False)
        test_preds, targs = learn.get_preds(dl=test_dl)
        mse_model = mse(test_preds, targs).item()
                
        # Calculate MSE baseline (mean project duration)
        Ybase = np.repeat(df_train['actual_duration'].mean(), df_test.shape[0])
        mse_base = mean_squared_error(df_test['actual_duration'].values, Ybase)

        # Relative MSE
        rel_mse = mse_model / mse_base if mse_base != 0 else np.inf

    else:
        # Add some hyperparameters
        if mdl_name in ['rf','xgb','ebm', 'en']:
            hyperparameters['random_state'] = seed
        if mdl_name in ['rf','xgb','ebm']:
            hyperparameters['n_jobs'] = int(os.environ.get("SLURM_CPUS_PER_TASK", "1")) #-1
        if mdl_name == 'en': # SV is a deterministic model, so no need to set the random state
            hyperparameters['max_iter'] = 2000 # Increase the default value to ensure convergence

        if mdl_name == "en": # add interaction terms: polynomial features (degree=2)
            poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            Xtrain_poly = poly.fit_transform(Xtrain)
            Xtest_poly = poly.transform(Xtest)
            # Fit the model with scaled data for all features
            scaler = StandardScaler()   
            Xtrain_scaled = scaler.fit_transform(Xtrain_poly)
            Xtest_scaled = scaler.transform(Xtest_poly)
        else:
            # Fit the model with scaled data for all features
            scaler = StandardScaler()   
            Xtrain_scaled = scaler.fit_transform(Xtrain)
            Xtest_scaled = scaler.transform(Xtest)
 
        model=mdl(**hyperparameters)
        model.fit(Xtrain_scaled,Ytrain) 
        Ypred=model.predict(Xtest_scaled)

        # Calculate MSE baseline (mean project duration)
        #Ybase = np.repeat(Ytrain.mean(), len(Ytest))
        dummy = DummyRegressor(strategy='mean')
        dummy.fit(Xtrain_scaled, Ytrain)
        Ybase = dummy.predict(Xtest_scaled)
        mse_base = mean_squared_error(Ytest, Ybase)
        # Calculate the MSE model
        mse_model = mean_squared_error(Ytest, Ypred)
        # Relative MSE
        rel_mse = mse_model / mse_base if mse_base != 0 else np.inf

    end_time = time.time()
    elapsed_time = round(end_time - start_time,0) # in seconds

    # Update the test results
    mse_test.append([project_name,mdl_name,lhs_i] + [mdl_lhs_params.loc[lhs_i,c] for c in mdl_lhs_params.columns] + [rel_mse,elapsed_time])

# ---------------- Save retults ----------------
# Save the results of the model selection
pd.DataFrame(mse_test,columns=['project','model','lhs'] + mdl_lhs_params.columns.tolist() + ['RelMSE','time']).to_csv(f"{path_results}/test_lhs_model_selection_{mdl_name}_project_{project_name}.csv",index=False)

