#!/usr/bin/python
# Script for Alatmira
# command: python rf1.py >> random_forest_log.txt

import sys
import os
import time
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


if len(sys.argv) != 3:
    sys.exit("Wrong number of arguments, e.g., <<python model_selection.py <model_name> <n_sp> <lhs_trial> >>")
else:
    mdl_name=sys.argv[1]
    n_sp=int(sys.argv[2])
    n_trial=int(sys.argv[3])
           
# Seed for reproducibility
seed=112358

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
else:
    sys.exit(f"Model {mdl_name} is not recognized.")

# Hyperparameters for the models
if os.path.exists(os.path.join('./model_selection',f"lhs_{mdl_name}_parameters.csv")):
    df_param=pd.read_csv(os.path.join('./model_selection',f"lhs_{mdl_name}_parameters.csv"))
else:
    sys.exit(f"File lhs_{mdl_name}_parameters.csv not found in ./model_selection folder.")

# Projects dataset for model selection
df_projects=pd.read_csv(os.path.join('./results','model_selection_projects.csv'))
df_selection=df_projects[df_projects['ModelSelection']] # Filter ModelSelection True
sp_values=df_selection['SP'].unique() # Unique values of SP

# Set the hyperparameters to explore
param_keys=list(df_param.columns)

# Test the combinations of hyperparameters over a subset of projects SPj
mse_test = []
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

    # Test all combinations of hiperparameters
    for i in df_param.index:

        start_time = time.time()

        # Train the model
        if mdl_name in ['rf','xgb','ebm']:
            model = mdl(n_jobs=-1,random_state=seed)
        elif mdl_name in ['en']:
            model = mdl(random_state=seed)
        else:
            model=mdl()
        model.set_params(**{df_param.loc[i,:].to_dict()})

        # Fit the model with scaled data for all features
        scaler = StandardScaler()   
        Xtrain_scaled = scaler.fit_transform(Xtrain)
        Xtest_scaled = scaler.transform(Xtest)
        model.fit(Xtrain_scaled,Ytrain)
        Ypred=model.predict(Xtest_scaled)
        
        # Calculate the MSE
        mse = mean_squared_error(Ytest, Ypred)

        end_time = time.time()
        elapsed_time = round(end_time - start_time,0) # in seconds

        # Update the test run
        mse_test.append([sp_values[n_sp],project] + df_param.loc[i,:].values.tolist() + [mse,elapsed_time])

# Save the results of the model selection
pd.DataFrame(mse_test,columns=['SP','Project'] + df_param.columns.tolist() + ['MSE','time']).to_csv(f"lhs{n_trial}_model_selection_{mdl_name}_{n_sp}.csv",index=False)




