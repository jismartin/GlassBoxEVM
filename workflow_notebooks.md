# Workflow - Notebooks

## [`montecarlo_simulation.ipynb`](montecarlo_simulation.ipynb)

**Objective:** Produce the experimental dataset that feeds model selection and evaluation.

- Loads the selected projects from `PSPLIB_sample/psplib_sample_metadata.csv`.
- Tests the Monte Carlo workflow on an individual project.
- Adds risk parameters using a beta-PERT distribution.
- Defines an EVM control point, for example `EV=15%`.
- Runs or validates Monte Carlo simulations from that control point.
- Generates the project splits: `model_selection/project_selection.csv`, `model_evaluation/project_evaluation.csv`.
- Generates the fixed run split: `montecarlo/test_runs.csv`.

## [`latinhypercube.ipynb`](latinhypercube.ipynb)

**Objective:** Create the experimental hyperparameter grid for the selection phase.

- Defines the hyperparameter spaces for `rf`, `xgb`, `ebm`, `sv`, `en`, and `fastai`.
- Generates 50 configurations per model using Latin Hypercube Sampling.
- Uses linear, log-uniform, and constant scales depending on the hyperparameter.
- Permutes the columns to build LHS combinations.
- Saves the CSV files in `model_selection/hyperparameters/lhs_{model}_parameters.csv`.

## [`ckecking.ipynb`](ckecking.ipynb)

**Objective:** Quality-control notebook for the dataset and intermediate artifacts.

- Verifies that the PSPLIB sample contains the expected 400 projects.
- Checks that the `.sm` files in `PSPLIB_sample/` match `psplib_sample_combined.csv`.
- Reviews the partition between selection and evaluation projects.
- Checks that there are 50 LHS configurations per model.
- Inspects selected Monte Carlo simulation CSV files and the size of `test_runs.csv`.
- Verifies that there are 40 projects for selection and 360 for evaluation.

## [`model_selection/slurm_scripts.ipynb`](model_selection/slurm_scripts.ipynb)

**Objective:** Prepare the large-scale hyperparameter selection run on the cluster.

- Reads the list of selection projects.
- Generates SLURM scripts to run hyperparameter selection on Altamira.
- Creates jobs by model and project for `rf`, `xgb`, `sv`, `en`, `ebm`, and `fastai`.
- Configures time limits, CPU resources, the Python environment, and the call to `model_selection/model_selection.py`.

## [`model_selection/fastai_model_selection_runs.ipynb`](model_selection/fastai_model_selection_runs.ipynb)

**Objective:** Complete FastAI selection results when they have not been generated on the cluster or when some runs are missing.

- Iterates over pending FastAI jobs.
- Detects which FastAI projects still do not have result CSV files.
- Runs `model_selection.py fastai {project_name}` locally for the missing projects.

## [`model_selection.ipynb`](model_selection.ipynb)

**Objective:** Choose the final LHS configuration for each model family.

- Loads selection results from `model_selection/results`.
- Validates that each model has complete results: 50 LHS configurations x 40 projects.
- Compares LHS configurations by model using `RelMSE`.
- Applies the Friedman test and Conover post-hoc test.
- Builds critical difference/ranking diagrams.
- Selects the hyperparameter configurations that are not significantly worse than the best one.
- Saves the summary of best configurations in `model_selection/selected_models_results.csv`.

## [`model_evaluation/slurm_scripts.ipynb`](model_evaluation/slurm_scripts.ipynb)

**Objective:** Prepare the large-scale final model evaluation on the projects not used for selection.

- Reads the 360 projects from `model_evaluation/project_evaluation.csv`.
- Defines the selected LHS for each model: `ebm: 24`, `rf: 14`, `xgb: 39`, `sv: 0`, `en: 36`, `fastai: 41`.
- Generates SLURM array scripts, one array per model.
- Each task evaluates one project with the model's final configuration.

## [`model_evaluation.ipynb`](model_evaluation.ipynb)

**Objective:** Obtain the final benchmark across models.

- Loads the final results from `model_evaluation/results`.
- Filters each model by its selected LHS.
- Compares overall performance across the 360 projects.
- Evaluates accuracy versus computational time.
- Repeats the comparison by subset: `J30`, `J60`, `J90`, `J120`.
- Applies the Friedman test and Conover post-hoc test.
- Calculates bootstrap intervals, effect sizes, win-rate matrices, and practical equivalence.
- Saves analysis tables in `model_evaluation/analysis`.

## [`topology_analysis.ipynb`](topology_analysis.ipynb)

**Objective:** Explain when and under which topological characteristics each model performs best.

- Loads final evaluation results and PSPLIB topological metadata.
- Joins model performance with indicators such as `SP`, `TF`, `CNC`, `OS`, etc.
- Explores the distribution and correlation of topological metrics.
- Analyzes how error/ranking changes according to project topology.
- Studies regions where each model wins, especially in the `SP` and `TF` planes.
- Runs statistical comparisons by topological cells.
- Saves derived tables in `model_evaluation/analysis`.
