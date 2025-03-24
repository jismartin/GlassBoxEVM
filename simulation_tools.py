import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy.stats as stats
import re
import os
from typing import Tuple
from network_tools import read_dag_file, add_risk_parameters


def pert_sample(optimistic: float, most_probable: float, pessimistic: float, elapsed_time: float =0, complete_percent: float =0,
                size: int =1000, lamb: float =4) -> np.ndarray:
    """
    Generate random samples from a PERT-like distribution.

    Parameters:
        optimistic (float): Optimistic estimate (O)
        most_probable (float): Most Probable estimate (M)
        pessimistic (float): Pessimistic estimate (P)
        elapsed_time (float): Elapsed time since the start of the activity
        size (int): Number of samples to generate
        lamb (float): Shape parameter (lower values = skewed distribution)

    Returns:
        numpy array of sampled durations

    """

    # Planned estimates
    a=optimistic
    b=most_probable
    c=pessimistic

    # Compute alpha and Beta parameters
    alpha = 1 + lamb * (b-a) / (c-a)
    beta = 1 + lamb * (c-b) / (c-a)

    # Generate samples from Beta distribution
    samples = stats.beta.rvs(alpha, beta, size=size)

    # Scale samples to the feasible range
    samples = a + samples * (c-a)
    return samples if complete_percent == 0 else elapsed_time + (1-complete_percent) * samples	


# Draw random durations for the activities following a PERT distribution
def draw_random_durations(G,pert_sample=pert_sample):
    
    nx.set_node_attributes(G,0,'duration')
    for n in list(G.nodes())[1:-1]: # Skip the start and end nodes
        G.nodes[n]['duration']=pert_sample(G.nodes[n]['Optimistic'], G.nodes[n]['MostLikely'], G.nodes[n]['Pessimistic'],size=1)[0]


# Compute the start time of activities based on the dependencies
def compute_times(G,time='duration'):
    # time: duration or meanduration
    start_time={n:0 for n in G.nodes()} # initialize the start time dictionary
    finish_time={n:0 for n in G.nodes()} # initialize the finish time dictionary
    for n in nx.topological_sort(G): # follow a list of nodes in topologically sorted order
        end=start_time[n] + G.nodes[n][time] # end time
        for j in G.neighbors(n):
            start_time[j]=np.max([start_time[j],end])
    for n in G.nodes():
        finish_time[n]=start_time[n]+G.nodes[n][time]
    return start_time,finish_time

def find_critical_path(G, time='duration'):
    # Compute the start and finish times
    start_time, finish_time = compute_times(G, time)  # Assume function is correctly implemented

    # Identify the unique end node using topological sorting
    topo_order = list(nx.topological_sort(G))
    end_node = topo_order[-1]  # The last node in topological order is the end node

    # Trace back from the end node using reverse topological order
    critical_path = [end_node]
    current = end_node
    while True:
        predecessors = list(G.predecessors(current))
        if not predecessors:
            break
        # Choose the predecessor that has the latest finish time that matches its successor's start time
        critical_predecessor = max(predecessors, key=lambda p: finish_time[p])
        if finish_time[critical_predecessor] != start_time[current]:
            break
        critical_path.append(critical_predecessor)
        current = critical_predecessor

    critical_path.reverse()  # Reverse to get the correct order
    return critical_path

# Monte-Carlo simulation of the project
def simulation(G,project_name,experiment,Nruns=10000,echo=False):
    sim_list=list() # list of dictionaries with runs
    if echo: print('Starting simulation...')
    for m in range(Nruns):
        # Draw durations
        draw_random_durations(G)
        # Save run
        sim={'duration'+str(i):G.nodes[i]['duration'] for i in G.nodes()}
        sim.update({'critical_path':find_critical_path(G,time='duration')})
        sim.update({'baseline_duration':np.max([t for t in compute_times(G,time='duration0')[1].values()])}) # baseline duration based on mean durations
        sim.update({'actual_duration':np.max([t for t in compute_times(G,time='duration')[1].values()])}) # final duration based on actual durations
        sim_list.append(sim)
        if echo: print(m)
    if echo: print('... end simulation')

    if not os.path.exists(os.path.join('./data/',project_name)):
        os.makedirs(os.path.join('./data/',project_name))
    pd.DataFrame.from_dict(sim_list).to_csv('./data/'+project_name+'/'+project_name+'_simulation_' + experiment+ '.csv')



def monte_carlo_simulation(G, project_file, experiment="e01", Nruns=10000, echo=False):

    G = read_dag_file(project_file)
    project_name = re.search(r'EV\d+', "dataset\\Set 1 SP\\EV401.rcp").group()
    add_risk_parameters(G,optim=0.8, mostprob=1.1, pessim=1.2)
    if echo:
        print('Simulation for ' + project_name +' ...')
    start_time = time.time()
    simulation(G.copy(),project_name=project_name, experiment="",Nruns=Nruns)
    if echo:
        print(f"Monte Carlo Simulation took {time.time() - start_time} seconds to run {Nruns} samples")


def evaluate_model(X, y, model, param_dist, project_name, model_name=None, 
                   n_splits=5, n_iter=10, random_seed=42, n_jobs=6, echo=True, save=True):
    """
    Evaluate a machine learning model using KFold cross-validation and RandomizedSearchCV for hyperparameter tuning.

    Parameters:
        X (pd.DataFrame): Feature data.
        y (pd.Series or array-like): Target variable.
        model: An instance of a machine learning model (e.g., RandomForestRegressor()).
        param_dist (dict): Dictionary of parameter distributions for tuning.
        project_name (str): Project name for building the output file path.
        model_name (str): Name for the model. Defaults to model.__class__.__name__ if None.
        n_splits (int): Number of folds for cross-validation.
        n_iter (int): Number of iterations for RandomizedSearchCV.
        random_seed (int): Seed for reproducibility.
        n_jobs (int): Number of jobs to run in parallel.

    Returns:
        list: List of mean squared errors for each fold.
    """
    if model_name is None:
        model_name = model.__class__.__name__
    
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    mse_folds = []
    start_time = time.time()
    
    for fold, (train_index, test_index) in enumerate(kfold.split(X)):
        if echo:
            print(f'Running fold {fold}...')
        search = RandomizedSearchCV(
            model,
            param_distributions=param_dist,
            n_iter=n_iter,
            n_jobs=n_jobs,
            random_state=random_seed
        )
        search.fit(X.iloc[train_index, :], y.iloc[train_index])
        y_pred = search.best_estimator_.predict(X.iloc[test_index, :])
        mse = mean_squared_error(y.iloc[test_index], y_pred)
        mse_folds.append(mse)
    
    total_time = time.time() - start_time
    if echo:
        print(f"{model_name} took {total_time:.2f} seconds to run {n_splits} folds")
    
    if save:
        results_df = pd.DataFrame(
            [mse_folds],
            columns=[f'mse{i+1}' for i in range(len(mse_folds))]
        )
        output_path = os.path.join('.', 'data', project_name, 
                               f"{project_name}_regression_model_selection_{model_name}.csv")
        results_df.to_csv(output_path, index=False)
    return mse_folds


