#!/usr/bin/python
# Script for Alatmira
# command: python rf1.py >> random_forest_log.txt

import sys
import os
#import itertools
import pandas as pd
from fastai.tabular.all import *
from fastai.metrics import mse
from sklearn.model_selection import train_test_split

# Hyperparameters of Neural Network
hyperparam={
    "layers": [ [100,50],[200,100],[300,200],[400,300]]
}

# Seed for reproducibility
#seed=112358

# Projects dataset for model selection
df_projects=pd.read_csv(os.path.join('./results','model_selection_projects.csv'))
df_selection=df_projects[df_projects['ModelSelection']] # Filter ModelSelection True

mse_test = []
for sp in df_selection['SP'].unique():

    # List of projects for training and testing
    for project in df_selection.loc[df_selection['SP'] == sp,'Project']:

        # Read the simulation data
        df = pd.read_csv(os.path.join('./results',f"{project}_simulation_e1.csv"),index_col=0)

        # Read test runs
        df_runs= pd.read_csv(os.path.join('./results',"test_runs.csv"))

        # Training and testing data
        df_train=df.loc[df.index.difference(df_runs['test_runs'].values)]
        df_test=df.loc[df_runs['test_runs'].values]

        # (1) Prepare training/validation set
        splits = RandomSplitter(valid_pct=0.1)(range_of(df_train)) # We assign the validation set to 10% of the training data
        to = TabularPandas(df_train, 
                procs=[Categorify,FillMissing,Normalize],
                cat_names = [],
                cont_names = [ col for col in df_train.columns if col.startswith('duration')],
                y_names='actual_duration',
                splits=splits)
        # Create DataLoaders, 'dls' will have 'train' and 'valid' DataLoaders
        dls = to.dataloaders(bs=64) # Try other batch sizes?

        # (2) Test all combinations of hiperparameters
        for layers in hyperparam['layers']:
            # (3) Train the model
            epochs = 100
            learn = tabular_learner(
                dls,
                layers=layers,
                metrics=mse,
                loss_func=MSELossFlat(), # Mean Squared Error as regression loss
                cbs=[EarlyStoppingCallback(monitor='valid_loss', patience=10)]
            )
            # Find the best learning rate
            lr_suggestion = learn.lr_find(show_plot=False)
            # and train the model
            learn.fit_one_cycle(epochs, lr_max=lr_suggestion)

            # (4) Evaluate the model on the test dataset
            test_dl = learn.dls.test_dl(df_test)
            test_preds = learn.get_preds(dl=test_dl)
            test_mse = mse(test_preds[0], tensor(df_test['actual_duration'].values)).item()

            # Save the results of the model selection       
            mse_test.append([sp,project,layers,mse])

# Save the results of the model selection
pd.DataFrame(mse_test,columns=['SP','Project','layers','MSE']).to_csv(f"model_selection_fastai.csv",index=False)

