import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy.stats as stats
from collections import Counter
import re
import os
from typing import Tuple


# Network tools


# Read a project file in Patterson's format and create the Directec Acyclic Graph (DAG)
def read_dag_file(file_path: str) -> nx.DiGraph:
    """
    Read a project file in Patterson's format and create the Directec Acyclic Graph (DAG).
    In Vanhoucke's dataset, the first line and the fourth line are empty, and empty lines are used to separate the data.
    
    Args:
        file_path (str): The path to the project file.

    Returns:
        nx.DiGraph: A directed acyclic graph representing a project network.
    """
    G=nx.DiGraph()
    try:
        with open(file_path, 'r') as file:
            # Read all non-empty lines
            lines = [line.strip() for line in file if line.strip()]
            
            # Read number of activities and resources
            num_activities, num_resources = map(int, lines[0].split())
            
            # Read resource availability
            resource_availability = list(map(int, lines[1].split()))
            
            # Read data for each activity
            for i in range(num_activities):
                data = lines[2 + i].split()
                activity_id = i + 1
                duration = int(data[0])
                G.add_node(activity_id, duration0=duration)
                required_resources = list(map(int, data[1:1 + num_resources]))
                num_successors = int(data[1 + num_resources])
                for successor in data[2 + num_resources:]:
                    G.add_edge(activity_id, int(successor))
        return G

    # try:
    #     with open(file_path, 'r') as file:
    #         # Fisrt line is empty
    #         file.readline()
    #         # Read number of activities and resources
    #         num_activities, num_resources = map(int, file.readline().strip().split())
    #         # Read resource availability
    #         resource_availability = list(map(int, file.readline().strip().split()))
    #         # Fourth line is empty
    #         file.readline()
    #         # Read data for each activity
    #         for activity_id in range(1, num_activities + 1):
    #             data = file.readline().strip().split()
    #             duration = int(data[0])
    #             G.add_node(activity_id, duration0=duration)
    #             required_resources = list(map(int, data[1:1 + num_resources]))
    #             num_successors = int(data[1 + num_resources])
    #             for successor in data[2 + num_resources:]:
    #                 G.add_edge(activity_id, int(successor))
    #     return G

    except Exception as e:
        print(f"Error reading the file: {e}")
        return None

# Add risk parameters to the nodes of a project network
def add_risk_parameters(G: nx.DiGraph, optim: float, mostprob: float, pessim: float) -> None:
    """
    Add risk parameters to the nodes of a project network.

    Args:
        G (nx.DiGraph): A directed acyclic graph representing a project network.
        optim (float): The optimistic factor for the risk.
        mostprob (float): The most probable factor for the risk.
        pessim (float): The pessimistic factor for the risk.
            
    Returns:    
        None: The function modifies the graph in place.
    """

    # parameters relative to baseline duration
    for n in G.nodes():
        G.nodes[n]['Optimistic']=G.nodes[n]['duration0']*optim
        G.nodes[n]['MostLikely']=G.nodes[n]['duration0']*mostprob
        G.nodes[n]['Pessimistic']=G.nodes[n]['duration0']*pessim
        G.nodes[n]['MeanDuration']=(G.nodes[n]['Optimistic']+4*G.nodes[n]['MostLikely']+G.nodes[n]['Pessimistic'])/6


# Compute the topological indicators for a given network (Vanhoucke, 2008)
def compute_topological_indicators(Go: nx.DiGraph) -> Tuple[float, float, float, float]:
    """
    Computes four topological indicators proposed by Vanhoucke of a directed graph.
    Asuming the graph is a directed acyclic graph (DAG) with dummy start and end nodes (Patterson's format).

    Args:
        G (nx.DiGraph): A directed acyclic graph representing a project network. 
        The start and end nodes are dummy activities.

    Returns:
        tuple[float, float, float, float]: A tuple containing four topological 
        indicators: the serial-parallel (SP), the activity distribution (AD),
        the short arcs (LA), and the topological flow (TF).
    """
    G=Go.copy()
    lorder = list(nx.topological_sort(G))
    # Remove start and end nodes
    G.remove_node(lorder[0])    
    G.remove_node(lorder[-1])

    lorder = list(nx.topological_sort(G))
    n = G.number_of_nodes() 
    lorder = list(nx.topological_sort(G))
    
    # Compute the Coefficient of Network Complexity (CNC)
    CNC = G.number_of_edges() / n if n > 0 else 0

    # Compute the Order Strnegth (OS)
    # Count the number of edges in the transitive closure (precedence relations)
    num_precedence_relations = nx.transitive_closure(G).number_of_edges()
    OS = num_precedence_relations / (n * (n - 1) / 2) if n > 1 else 0


    # Compute progressive levels (PL)
    Plevels = {node: 0 for node in G.nodes}
    for node in lorder:  
        Plevels[node] = 1 + max((Plevels[p] for p in G.predecessors(node)), default=0)
    m = max(Plevels.values(), default=0)

    # Compute regressive levels (RL)
    Rlevels = {node: 0 for node in G.nodes}
    for node in list(reversed(lorder)):
        Rlevels[node] = min((Rlevels[s] for s in G.successors(node)), default=m+1) - 1


    # Count occurrences of each progressive level
    Wlevels_counts = Counter(Plevels.values())
    Wlevels = np.array([Wlevels_counts.get(a, 0) for a in range(1, m + 1)])
    #Wlevels = np.array(list(Wlevels_counts.values()))
    D = np.sum(Wlevels[:-1] * Wlevels[1:]) if len(Wlevels) > 1 else 0

    # Serial-parallel indicator (I2-SP)
    SP = (m - 1) / (n - 1) if n > 1 else 1

    # Activity distribution indicator (I3-AD)
    W_mean = Wlevels.mean() if len(Wlevels) > 0 else 0
    AD = np.abs(Wlevels - W_mean).sum() / (2 * (m - 1) * (W_mean - 1)) if (m > 1) and (m < n) else 0

    # Compute short arcs indicator (I4-LA)
    lengths_arcs = [Plevels[j] - Plevels[i] for i, j in G.edges ]
    lengths_counts = Counter(lengths_arcs)
    n1 = lengths_counts.get(1, 0)  # number of arcs of length 1
    # w1 = number of activities at progressive level 1
    w1 = Wlevels[0] if len(Wlevels) > 0 else 0
    if D == (n - w1):
        LA = 1
    else:
        LA = (n1 - n + w1) / (D - n + w1)

    # Compute I5
    total_arcs = sum(lengths_counts.values())  # |A|
    if total_arcs == n - w1:
        I5 = 1
    else:
        weighted_sum = sum( lengths_counts.get(l, 0) * ((m - l - 1) / (m - 2)) 
            for l in range(2, m) if (m - 2) > 0)
    I5 = (weighted_sum + n1 - n + w1) / (total_arcs - n + w1) if (total_arcs - n + w1) != 0 else 0

    # Topological flow indicator (I6-TF)
    TF = np.sum(np.fromiter((Rlevels[node] - Plevels[node] for node in Plevels), dtype=int)) / ((m - 1) * (n - m)) if (m > 1) and (m < n) else 0

    return n, CNC, OS, SP, AD, LA, I5, TF

# Visualize the DAG using a hierarchical layout
def visualize_dag(G: nx.DiGraph, figsize: Tuple[int, int] = (8, 6)) -> None:
    """
    Visualize a Directed Acyclic Graph (DAG) with a hierarchical layout.

    Args:
    G (networkx.DiGraph): A directed acyclic graph.
    figsize (tuple): Figure size (width, height).
    """
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("The provided graph is not a Directed Acyclic Graph (DAG).")

    # Create a copy of the graph to avoid modifying the original
    G0=G.copy()

    # Compute a hierarchical layout for better visualization of DAGs
    lorder = list(nx.topological_sort(G0))
    
    # Compute progressive levels (PL)
    Plevels = {node: 0 for node in G0.nodes}
    for node in lorder[1:-1]:  # Skip start and end
        Plevels[node] = 1 + max((Plevels[p] for p in G0.predecessors(node)), default=0)
    Plevels[max(G0.nodes())] = max(Plevels.values()) + 1
    nx.set_node_attributes(G0, Plevels, "subset")
    pos = nx.drawing.layout.multipartite_layout(G0, subset_key='subset')

    # Draw the graph
    plt.figure(figsize=figsize)
    nx.draw(G0, pos, with_labels=True, node_color="lightblue", alpha=0.7, edge_color="gray", arrows=True)
    plt.show()


# Simulation tools

# Generate random samples from a PERT-like distribution
def pert_sample(optimistic: float, most_probable: float, pessimistic: float, elapsed_time: float =0, complete_percent: float =0,
                size: int =1000, lamb: float =4) -> np.ndarray:
    """
    Generate random samples from a PERT-like distribution.

    Args:
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
def draw_random_durations(G: nx.DiGraph, pert_sample=pert_sample):
    """
    Draw random durations for the activities following a PERT distribution.
    The function modifies the 'duration' attribute of each node in the graph.
        
    Args:
        G (nx.DiGraph): A directed acyclic graph representing a project network.
        pert_sample (function): Function to sample durations from a PERT distribution.
        
    Returns:
        None: The function modifies the graph in place.
    """    
    nx.set_node_attributes(G,0,'duration')
    for n in list(G.nodes())[1:-1]: # Skip the start and end nodes
        G.nodes[n]['duration']=pert_sample(G.nodes[n]['Optimistic'], G.nodes[n]['MostLikely'], G.nodes[n]['Pessimistic'],size=1)[0]


# Compute the start time of activities based on the dependencies
def compute_times(G: nx.DiGraph, time='duration'):
    """
    Compute the start and finish times of activities in a directed acyclic graph (DAG).
    
    Args:
        G (nx.DiGraph): A directed acyclic graph representing a project network.
        time (str): The attribute name for the duration of the activities. Default is 'duration'.
        
    Returns:
        tuple: A tuple containing two dictionaries:
            - start_time: A dictionary mapping each node to its start time.
            - finish_time: A dictionary mapping each node to its finish time.
    """
    
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


# Find the critical path in a directed acyclic graph (DAG)
def find_critical_path(G, time='duration'):
    """
    Find the critical path in a directed acyclic graph (DAG) using topological sorting.
        
    Args:
        G (nx.DiGraph): A directed acyclic graph representing a project network.
        time (str): The attribute name for the duration of the activities. Default is 'duration'.
            
    Returns:
        list: A list of nodes representing the critical path in the graph.
    """
    
    # Compute the start and finish times
    start_time, finish_time = compute_times(G, time)  

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
def simulation(G: nx.DiGraph, project_name: str, experiment: str,Nruns: int=10000,echo: bool=False):
    """
    Monte Carlo simulation of the project using a PERT distribution.

    Args:
        G (nx.DiGraph): A directed acyclic graph representing a project network.
        project_name (str): Name of the project for saving results.
        experiment (str): Experiment identifier for saving results.
        Nruns (int): Number of simulation runs.
        echo (bool): Whether to print progress messages.

    Returns:
        None: The function saves the simulation results to a CSV file.
    """

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

    pd.DataFrame.from_dict(sim_list).to_csv('./results/'+project_name+'_simulation_' + experiment+ '.csv')




def evaluate_model(X, y, model, param_dist, project_name, model_name=None, 
                   n_splits=5, n_iter=10, random_seed=42, n_jobs=6, echo=True, save=True):
    """
    Evaluate a machine learning model using KFold cross-validation and RandomizedSearchCV for hyperparameter tuning.

    Args:
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


