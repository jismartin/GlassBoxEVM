#!/usr/bin/python
# Script for Alatmira
# command: python rf1.py >> random_forest_log.txt

import sys
import os
import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from fastai.tabular.all import *
from fastai.metrics import mse
from sklearn.model_selection import train_test_split

import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Model selection script for Altamira')

# Arguments
parser.add_argument('model_name', type=str, choices=['rf', 'xgb', 'ebm', 'sv', 'en', 'fastai'],
                    help='Short name of the model (rf, xgb, ebm, sv, en, fastai)')
parser.add_argument('lhs_trial', type=int, help='Index of the LHS trial')
parser.add_argument('n_sp', type=int, help='Index of the SP (subset of projects)')
args = parser.parse_args()

# Get parsed values
mdl_name = args.model_name
n_trial = args.lhs_trial
n_sp = args.n_sp

# Seed for reproducibility
seed=112358

# Path for the results 
path_model_selection='./model_selection'
path_montecarlo='./montecarlo'

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
    mdl = tabular_learner
else:
    sys.exit(f"Model {mdl_name} is not recognized.")

# Hyperparameters for the models
if os.path.exists(os.path.join(path_model_selection,f"lhs{n_trial}_{mdl_name}_parameters.csv")):
    df_param=pd.read_csv(os.path.join(path_model_selection,f"lhs{n_trial}_{mdl_name}_parameters.csv"),index_col=0)
else:
    sys.exit(f"File lhs{n_trial}_{mdl_name}_parameters.csv not found in {path_model_selection} folder.")

# Projects dataset for model selection
df_projects=pd.read_csv(os.path.join(path_montecarlo,'model_selection_projects.csv'))
df_selection=df_projects[df_projects['ModelSelection']] # Filter ModelSelection True
sp_values=df_selection['SP'].unique() # Unique values of SP
sp_values.sort()

# Test the combinations of hyperparameters over a subset of projects SPj
mse_test = []
for project in df_selection.loc[df_selection['SP'] == sp_values[n_sp],'Project']:

    # Read the simulation data
    df = pd.read_csv(os.path.join(path_montecarlo,f"{project}_simulation_e1.csv"),index_col=0)

    # Read test fold
    df_runs= pd.read_csv(os.path.join(path_montecarlo,"test_runs.csv"))

    # Training and testing data
    df_train=df.loc[df.index.difference(df_runs['test_runs'].values)]
    df_test=df.loc[df_runs['test_runs'].values]
    if mdl_name == 'fastai':
        # Fastai requires a different approach for training and testing data
        splits = RandomSplitter(valid_pct=0.1)(range_of(df_train)) # We assign the validation set to 10% of the training data
        to = TabularPandas(df_train, 
                procs=[Categorify,FillMissing,Normalize],
                cat_names = [],
                cont_names = [ col for col in df_train.columns if col.startswith('duration')],
                y_names='actual_duration',
                splits=splits)
        # Create DataLoaders, 'dls' will have 'train' and 'valid' DataLoaders
        dls = to.dataloaders(bs=64) # Try other batch sizes?
    else:
        Xtrain=df_train.loc[:,[ col for col in df_train.columns if col.startswith('duration')]] 
        Ytrain=df_train['actual_duration']
        Xtest=df_test.loc[:,[ col for col in df_train.columns if col.startswith('duration')]] 
        Ytest=df_test['actual_duration']

    # Test all combinations of hiperparameters
    for i in df_param.index:

#        print(f"Running {mdl_name} for {project} with {i}-LHShyperparameters")

        # For comuting the run time
        start_time = time.time()

        # Load the hyperparameters
        hyperparameters = df_param.iloc[[i]].to_dict(orient='records')[0] # Using `.loc[i, :]` returns a Series, which may upcast ints to floats — use `.iloc[[i]]` for dtype-safe row extraction.

        if mdl_name in ['rf','xgb','ebm','en']: # Fix the seed when it's possible
            hyperparameters['random_state'] = seed
        # SV is a deterministic model, so no need to set the random state

        if mdl_name in ['rf','xgb','ebm']: # Use all the cores for parallel processing in ensemble models
            hyperparameters['n_jobs'] = -1

        if mdl_name == 'xgb': # Set the learning rate for XGBoost according to the number of estimators
            hyperparameters['learning_rate'] = 1/np.sqrt(hyperparameters['n_estimators'])

        # Load and train the model
        if mdl_name == 'fastai':
            # Fastai requires a different approach for model creation
            layers=[]
            layers.append(int((hyperparameters['layer1']//5)*5)) # Round to the nearest multiple of 5
            layers.append(int((hyperparameters['layer2']//5)*5)) # Round to the nearest multiple of 5
            epochs = 100
            model = mdl(
                dls,
                layers=layers,
                metrics=mse,
                #config=tabular_config(ps=hyperparameters['ps']), # Dropout for embedding layers
                wd=hyperparameters['wd'], # Weight decay
                loss_func=MSELossFlat(), # Mean Squared Error as regression loss
                cbs=[EarlyStoppingCallback(monitor='valid_loss', patience=10)]
            )
            # Find the best learning rate
            lr_suggestion = model.lr_find(show_plot=False)
            # and train the model
            model.fit_one_cycle(epochs, lr_max=lr_suggestion)
            # Evaluate the model on the test dataset
            test_dl = model.dls.test_dl(df_test)
            test_preds = model.get_preds(dl=test_dl)
            test_mse = mse(test_preds[0], tensor(df_test['actual_duration'].values)).item()

        else:
            model=mdl(**hyperparameters)
            # Fit the model with scaled data for all features
            scaler = StandardScaler()   
            Xtrain_scaled = scaler.fit_transform(Xtrain)
            Xtest_scaled = scaler.transform(Xtest)
            model.fit(Xtrain_scaled,Ytrain)
            Ypred=model.predict(Xtest_scaled)            
            # Calculate the MSE
            test_mse = mean_squared_error(Ytest, Ypred)

        # Elapsed time for the test run
        end_time = time.time()
        elapsed_time = end_time - start_time # in seconds

        # Update the test run
        mse_test.append([sp_values[n_sp],project,i] + [df_param.loc[i,c] for c in df_param.columns] + [test_mse,elapsed_time])

# Save the results of the model selection
pd.DataFrame(mse_test,columns=['SP','Project','lhs_param'] + df_param.columns.tolist() + ['MSE','time']).to_csv(os.path.join(path_model_selection,f"lhs{n_trial}_model_selection_{mdl_name}_sp{n_sp}.csv"),index=False)
