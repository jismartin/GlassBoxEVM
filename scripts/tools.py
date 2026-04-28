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

## Graph attributes (for each activity-node)
#
## PSPLIB dataset
# pert_duration
# cost: variable cost (proportional to total effort)
#
## beta-PERT distribution
# Optimistic: optimistic duration
# MostLikely: most likely duration
# Pessimistic: pessimistic duration
# MeanDuration: mean duration
#
## Control point (at EV)
# duration@: duration at EV
# ev@: plan cost incurred up to EV
# done%: percentage of completion
#
## Simulation
# duration: duration sampled from beta-PERT distribution

# Read a project file in PSPLIB format and create the Directec Acyclic Graph (DAG)
def read_sm_file(file_path: str) -> nx.DiGraph:
    """
    Read a project file in PSPLIB format and create the Directec Acyclic Graph (DAG).
    Update PSPLIB attributes of the nodes.

    Args:
        file_path (str): The path to the project file.

    Returns:
        nx.DiGraph: A directed acyclic graph representing a project network.
    """

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        raw = [ln.rstrip("\n") for ln in f]

    # Get number of jobs
    for line in raw:
        if "jobs (incl. supersource/sink ):" in line:
            n_jobs = int(line.rstrip("\n").split()[-1])


    G=nx.DiGraph()
    
    # Find the lines with precedence relations and requests/durations
    l_precedence = -1
    l_requests = -1
    for (i,line) in enumerate(raw):
        if "PRECEDENCE RELATIONS:" in line:
            l_precedence = i + 2
        if "REQUESTS/DURATIONS:" in line:
            l_requests = i + 3
    if (l_precedence == -1) or (l_requests == -1):
        raise ValueError("Line with 'PRECEDENCE OR REQUESTS/DURATIONS:' not found.")

    # Add nodes and edges to the graph
    for i in range(n_jobs):
        node=int(raw[l_precedence + i].split()[0])
        duration=int(raw[l_requests + i].split()[2])
        
        # There is no activity cost data in PSPLIB dataset
        # We assume that resources are similarly priced and (variable) costs are proportional to total effort (number of resource units)
        cost = sum([int(r) for r in raw[l_requests + i].split()[3:]])  
        
        G.add_node(node,pert_duration=duration,cost=cost)
        if int(raw[l_precedence + i ].split()[2]) != 0:
            for successor in raw[l_precedence + i].split()[3:]:
                G.add_edge(node, int(successor))
    
    return G

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

# Add risk parameters to the nodes of a project network
def add_risk_parameters(G: nx.DiGraph, optim: float, mostprob: float, pessim: float) -> None:
    """
    Add risk parameters to the nodes of a project network.
    Update the beta-PERT attributes of the nodes.

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
        G.nodes[n]['Optimistic']=G.nodes[n]['pert_duration']*optim
        G.nodes[n]['MostLikely']=G.nodes[n]['pert_duration']*mostprob
        G.nodes[n]['Pessimistic']=G.nodes[n]['pert_duration']*pessim
        G.nodes[n]['MeanDuration']=(G.nodes[n]['Optimistic']+4*G.nodes[n]['MostLikely']+G.nodes[n]['Pessimistic'])/6
        G.nodes[n]['duration'] = 0 # duration sampled form beta-PERT

# Generate random samples from a beta-PERT distribution
def beta_pert_sample(optimistic: float, most_probable: float, pessimistic: float, elapsed_time: float =0, complete_percent: float =0,
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

# Draw random durations for the activities following a beta-PERT distribution
def draw_random_durations(G,control=False,beta_pert_sample=beta_pert_sample):
    """
    Draw random durations for the activities following a beta-PERT distribution.
    Update duration attribute of the nodes.

    Args:
        G (nx.DiGraph): A directed acyclic graph representing a project network.
        control (bool, optional): Whether to use control point attributes. Defaults to False.
        beta_pert_sample (callable, optional): Function to sample from the beta-PERT distribution. Defaults to beta_pert_sample.
    """
    
    nx.set_node_attributes(G,0,'duration')
    if control is False:
        for n in [j for j in G.nodes() if G.nodes[j]['pert_duration']>0]: # remove dummy nodes (start and end)
            G.nodes[n]['duration']=beta_pert_sample(G.nodes[n]['Optimistic'], G.nodes[n]['MostLikely'], G.nodes[n]['Pessimistic'],size=1)[0]
    else:
        for n in [j for j in G.nodes() if G.nodes[j]['pert_duration']>0]: # remove dummy nodes (start and end): 

            if G.nodes[n]['done%']>=1: # completed activity
                G.nodes[n]['duration'] = G.nodes[n]['duration@']
                
            elif G.nodes[n]['done%']>0 and G.nodes[n]['done%'] <1: # ongoing activity
                G.nodes[n]['duration']=beta_pert_sample(G.nodes[n]['Optimistic'], G.nodes[n]['MostLikely'], G.nodes[n]['Pessimistic'],
                                                   elapsed_time=G.nodes[n]['duration@'],
                                                   complete_percent=G.nodes[n]['done%'],
                                                   size=1)[0]

            else: # not started activity
                G.nodes[n]['duration']=beta_pert_sample(G.nodes[n]['Optimistic'], G.nodes[n]['MostLikely'], G.nodes[n]['Pessimistic'],size=1)[0] 

# Compute the start and finish time of activities based on the dependencies
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

# Evaluate project state at a particular earned value (EV)
def evaluate_at_EV(G,EV,Ti=0,N=10000):
    """
    Evaluate project status at a particular earned value (EV).
    Update the control point attributes of the nodes.
    
    Args:
        G (nx.DiGraph): A directed acyclic graph representing a project network.
        EV (float): The earned value to evaluate the project status.
        Ti (int, optional): The initial time to start the evaluation. Defaults to 0.
        N (int, optional): The number of points to sample in the evaluation. Defaults to 1000.

    """
    # Compute start and finish times based on the precedence relations and mean durations
    # initialize the duration to the mean duration
    for n in G.nodes():
        G.nodes[n]['duration']=G.nodes[n]['MeanDuration'] 
    start_time,finish_time =compute_times(G, time='duration')

    # Initialize DataFrame for earned value (EV) analysis
    df_ev=pd.DataFrame.from_dict({n:n for n in G.nodes()}, orient='index', columns=['node'])
    for col in ['cost','MeanDuration','duration']:
        df2=pd.DataFrame.from_dict(nx.get_node_attributes(G, col), orient='index', columns=[col])
        df_ev=pd.concat([df_ev,df2], axis=1)
    df_ev=pd.concat([df_ev, pd.DataFrame.from_dict(start_time, orient='index', columns=['t_start'])], axis=1)
    df_ev=pd.concat([df_ev, pd.DataFrame.from_dict(finish_time, orient='index', columns=['t_end'])], axis=1)
    df_ev['duration@']=0.0
    df_ev['ev@']=0.0
    df_ev['done%']=0.0

    # Matrix representation
    A=df_ev.values 
    c={'node':0, 'cost':1, 'MeanDuration':2, 'duration':3, 't_start':4, 't_end':5, 'duration@':6, 'ev@':7, 'done%':8}

    # Iterative approach (increase N to get more decimal precission)
    for T in np.linspace(Ti,df_ev['t_end'].max(),N):

        # Not started activities
        not_started = T<=A[:,c['t_start']]
        A[not_started,c['duration@']]=0
        A[not_started,c['ev@']]=0

        # Ended activities
        ended = T>=A[:,c['t_end']]
        A[ended,c['duration@']]=A[ended,c['duration']] # Actual duration
        A[ended,c['ev@']]=A[ended,c['MeanDuration']]*A[ended,c['cost']] # plan cost
        A[ended,c['done%']]=1.0 # 100% done

        # Running activities
        running = (A[:,c['t_start']]<T) & (A[:,c['t_end']]>T)
        completion_rate = (T-A[running,c['t_start']]) / (A[running,c['t_end']]-A[running,c['t_start']])
        A[running,c['duration@']] = A[running,c['duration']] * completion_rate # actual duration at EV
        A[running,c['ev@']] = A[running,c['MeanDuration']] * A[running,c['cost']] * completion_rate # plan cost incurred up to EV
        A[running,c['done%']] = completion_rate # % of the activity done
        if A[:,c['ev@']].sum()/(A[:,c['MeanDuration']]*A[:,c['cost']]).sum()>=EV:
            break;

    df_ev.loc[:,'duration@']=A[:,c['duration@']]
    df_ev.loc[:,'ev@']=A[:,c['ev@']]
    df_ev.loc[:,'done%']=A[:,c['done%']]
    
    # Update Graph
    for n in df_ev.index:
        G.nodes[n]['duration@'] = df_ev.loc[n, 'duration@']
        G.nodes[n]['ev@'] = df_ev.loc[n, 'ev@']
        G.nodes[n]['done%'] = df_ev.loc[n, 'done%']
    
# Find the critical path in the project
def find_critical_path(G, time='duration'):
    # Compute the start and finish times
    start_time, finish_time = compute_times(G, time)  

    # Identify the unique end node using topological sorting
    topo_order = list(nx.topological_sort(G))
    end_node = topo_order[-1]  # The last node in topological order is the end node

    # Trace back from the end node using reverse topological order
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
def simulation(
    G: nx.DiGraph,
    project_name: str,
    experiment: str,
    control: bool = False,
    Nruns: int = 10000,
    echo: bool = False,
    output_dir: str = './montecarlo',
):
    """
    Monte Carlo simulation of the project using a PERT distribution.

    Args:
        G (nx.DiGraph): A directed acyclic graph representing a project network.
        project_name (str): Name of the project for saving results.
        experiment (str): Experiment identifier for saving results.
        control (bool): Whether to use control point for starting the simulation.
        Nruns (int): Number of simulation runs.
        echo (bool): Whether to print progress messages.
        output_dir (str): Destination directory for the output CSV file.
            Defaults to './montecarlo'.

    Returns:
        None: The function saves the simulation results to a CSV file.
    """

    sim_list=list() # list of dictionaries with runs
    if echo: print(f'Starting simulation of {project_name}...')
    for m in range(Nruns):
        # Draw durations
        draw_random_durations(G,control=control)
        # Save run
        sim={'duration'+str(i):G.nodes[i]['duration'] for i in G.nodes()}
        #sim.update({'critical_path':find_critical_path(G,time='duration')})
        sim.update({'baseline_duration':np.max([t for t in compute_times(G,time='MeanDuration')[1].values()])}) # baseline duration based on mean durations
        sim.update({'actual_duration':np.max([t for t in compute_times(G,time='duration')[1].values()])}) # final duration based on actual durations
        sim_list.append(sim)
        #if echo: print(m)
    #if echo: print('... end simulation')

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, project_name + '_simulation_' + experiment + '.csv')
    pd.DataFrame.from_dict(sim_list).to_csv(output_path)




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


