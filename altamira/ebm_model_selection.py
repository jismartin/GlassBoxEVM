#!/usr/bin/python
# Script for Alatmira
# command: python rf1.py >> random_forest_log.txt

import sys
import os
import itertools
import pandas as pd
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

if len(sys.argv) != 4:
    sys.exit("Wrong number of arguments, e.g., <<python ebm_model_selection.py n_sp p_int p_lr >>")
else:
    n_sp=int(sys.argv[1])
    p_int=float(sys.argv[2])
    p_lr=float(sys.argv[3])

# Seed for reproducibility
seed=112358

# Hiperparameters of Explainable Boosting
#hyperparam={"interactions": [0,0.7,0.9],
#            "learning_rate": [0.01,0.04,0.1]}

# Projects dataset for model selection
df_projects=pd.read_csv(os.path.join('./results','model_selection_projects.csv'))
df_selection=df_projects[df_projects['ModelSelection']] # Filter ModelSelection True
sp_values=df_selection['SP'].unique() # Unique values of SP

# Test a specific combination of hyperparameters over a subset of projects SPj

mse_test = []
# List of projects for training and testing
for project in df_selection.loc[df_selection['SP'] == sp_values[n_sp],'Project']:

    # Read the simulation data
    df = pd.read_csv(os.path.join('./results',f"{project}_simulation_e1.csv"),index_col=0)

    # Read test runs
    df_runs= pd.read_csv(os.path.join('./results',"test_runs.csv"))

    # Training data
    df_train=df.loc[df.index.difference(df_runs['test_runs'].values)]
    Xtrain=df_train.iloc[:,:32] # 32 features correspondint to the 32 activities of the project
    Ytrain=df_train.iloc[:,-1] # Last column is the objective function: the actual duration of the project
    
    # Testing data
    df_test=df.loc[df_runs['test_runs'].values]
    Xtest=df_test.iloc[:,:32]
    Ytest=df_test.iloc[:,-1]

    # Train the model
    model = ExplainableBoostingRegressor(n_jobs=-1,random_state=seed,
                interactions=p_int,learning_rate=p_lr)

    # Fit the model with scaled data for all features
    scaler = StandardScaler()   
    Xtrain_scaled = scaler.fit_transform(Xtrain)
    Xtest_scaled = scaler.transform(Xtest)
    model.fit(Xtrain_scaled,Ytrain)
    Ypred=model.predict(Xtest_scaled)
    
    # Calculate the MSE
    mse = mean_squared_error(Ytest, Ypred)
    # Update the test runs
    mse_test.append([sp_values[n_sp],project,p_int,p_lr,mse])

# Save the results of the model selection
pd.DataFrame(mse_test,columns=['SP','Project','interactions','learning_rate','MSE']).to_csv(f"model_selection_ebm_{n_sp}_{p_int}_{p_lr}.csv",index=False)




