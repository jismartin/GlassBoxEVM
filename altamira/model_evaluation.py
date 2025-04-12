#!/usr/bin/python
# Script for Alatmira
# command: python rf1.py >> random_forest_log.txt

import sys
import os
import itertools
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


if len(sys.argv) != 2:
    sys.exit("the name of SP batch is required as an argument, e.g., <<python model_selection.py sp_number>>")
            
# Hyperparameters of Random Forest
hyperparam_rf={"n_estimators": 1000, "max_depth": None}

# Hiperparameters of XGBoost
hyperparam_xgb={"n_estimators": 1000, "learning_rate": 0.1}

# Hiperparameters of Explainable Boosting
hyperparam_ebm={"interactions": [0,0.7,0.9],
            "learning_rate": [0.01,0.04,0.1]}

# Hiperparameters of Support Vector Regressor (kernel=rfb)
hyperparam_svr={"C": 100, "gamma": 'scale'}

# Hiperparameters of Elastic Net
hyperparam_en={"alpha": 0.01,"l1_ratio": 0.1}

# Seed for reproducibility
seed=112358

# Name of the model
mdl_name=sys.argv[1]

if mdl_name == 'rf':
    hyperparam=hyperparam_rf
    mdl = RandomForestRegressor
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
else:
    sys.exit(f"Model {mdl_name} is not recognized.")

# Projects dataset for model selection
df_projects=pd.read_csv(os.path.join('./results','model_selection_projects.csv'))
df_selection=df_projects[df_projects['ModelSelection']] # Filter ModelSelection True

# Set the hyperparameters to explore
param_keys=list(hyperparam.keys())
param_values=list(hyperparam.values())

mse_test = []
for sp in df_selection['SP'].unique():

    # List of projects for training and testing
    for project in df_selection.loc[df_selection['SP'] == sp,'Project']:

        # Read the simulation data
        df = pd.read_csv(os.path.join('./results',f"{project}_simulation_e1.csv"),index_col=0)
        #df = pd.read_csv('./results',f"{project}_simulation_e1.csv"),index_col=0)

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
        for values in itertools.product(param_values[0],param_values[1]):
            # Train the model
            if mdl_name in ['rf','xgb','ebm']:
                model = mdl(n_jobs=-1,random_state=seed)
            elif mdl_name in ['en']:
                model = mdl(random_state=seed)
            else:
                model=mdl()
            model.set_params(**{param_keys[0]:values[0],param_keys[1]:values[1]})

            # Fit the model with scaled data for all features
            scaler = StandardScaler()   
            Xtrain_scaled = scaler.fit_transform(Xtrain)
            Xtest_scaled = scaler.transform(Xtest)
            model.fit(Xtrain_scaled,Ytrain)
            Ypred=model.predict(Xtest_scaled)
            
            # Calculate the MSE
            mse = mean_squared_error(Ytest, Ypred)
            # Update the test runs
            mse_test.append([sp,project,values[0],values[1],mse])
# Save the results of the model selection
pd.DataFrame(mse_test,columns=['SP','Project',param_keys[0],param_keys[1],'MSE']).to_csv(f"model_selection_{mdl_name}.csv",index=False)




