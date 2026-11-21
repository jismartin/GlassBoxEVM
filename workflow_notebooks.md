# Workflow: notebooks and scripts

The steps are listed in execution order. Paths are relative to the repository root, which is also the working directory, except for `PSPLIB_sample/psplib_sample_topology.ipynb`, which runs from `PSPLIB_sample/`.

## [`PSPLIB_sample/psplib_sample_topology.ipynb`](PSPLIB_sample/psplib_sample_topology.ipynb)

**Objective:** Describe the PSPLIB sample and validate its topological descriptors.

- Reads the PSPLIB instances of the sample.
- Computes the topological indicators (#Act, CNC, OS, SP, AD, LA, I5, TF) for one project and checks them against `psplib_sample_metadata.csv`.

## [`montecarlo_simulation.ipynb`](montecarlo_simulation.ipynb)

**Objective:** Produce the simulated dataset used for model selection and evaluation.

- Loads the sampled projects from `PSPLIB_sample/psplib_sample_metadata.csv`.
- Tests the Monte Carlo workflow on one project.
- Adds beta-PERT risk parameters (0.8d, 1.1d, 1.2d) to each activity.
- Sets the EVM control point at `EV=15%` of the budget at completion.
- Runs the Monte Carlo simulation (10,000 scenarios per project) from the control point and stores the outputs in `montecarlo/`.
- Writes the project splits: `model_selection/project_selection.csv` (40 projects) and `model_evaluation/project_evaluation.csv` (360 projects).
- Writes the fixed test indices: `montecarlo/test_runs.csv` (2,000 of the 10,000 scenarios).

## [`latinhypercube.ipynb`](latinhypercube.ipynb)

**Objective:** Create the hyperparameter configurations for model selection.

- Defines the hyperparameter spaces for `rf`, `xgb`, `ebm`, `sv`, `en` and `fastai`.
- Generates 50 configurations per model by Latin Hypercube Sampling, with linear, log-uniform or constant scales depending on the hyperparameter.
- Saves them to `model_selection/hyperparameters/lhs_{model}_parameters.csv`.

## [`checking.ipynb`](checking.ipynb)

**Objective:** Quality control of the dataset and intermediate files.

- Verifies that the PSPLIB sample contains 400 projects and that the `.sm` files match `psplib_sample_combined.csv`.
- Checks the split into 40 selection and 360 evaluation projects.
- Checks that there are 50 LHS configurations per model.
- Inspects some simulation files and the size of `test_runs.csv`.

## [`model_selection/slurm_scripts.ipynb`](model_selection/slurm_scripts.ipynb)

**Objective:** Prepare the hyperparameter selection runs on the Altamira cluster.

- Generates one SLURM script per model and selection project for `rf`, `xgb`, `sv`, `en` and `ebm`; each job runs the 50 LHS configurations with `model_selection/model_selection.py <model> <project>`. `model_selection/sbatch_model_jobs.sh` submits the jobs of one model.
- Generates one SLURM array per selection project for `fastai`, with one single-core task per LHS configuration (`model_selection.py fastai <project> --lhs <k>`).
- fastai arrays are submitted one PSPLIB set at a time with `bash model_selection/submit_fastai_wave.sh <SET>`. `python3 model_selection/merge_fastai_results.py` merges the per-configuration outputs into one CSV per project and lists any missing runs.

## [`model_selection.ipynb`](model_selection.ipynb)

**Objective:** Retain one configuration per model family.

- Loads the selection results from `model_selection/results/` and checks that each model has 50 configurations × 40 projects.
- Ranks the configurations within each project by `RelMSE`.
- Applies the Friedman test and the Conover post-hoc test, and draws critical-difference diagrams.
- Retains the configuration with the best mean rank, using rank variability as a secondary criterion.
- Saves the retained configurations to `model_selection/selected_models_results.csv`.

## [`model_evaluation/slurm_scripts.ipynb`](model_evaluation/slurm_scripts.ipynb)

**Objective:** Prepare the final evaluation on the 360 evaluation projects.

- Reads the projects from `model_evaluation/project_evaluation.csv`.
- Uses the retained configurations: `ebm: 24`, `rf: 14`, `xgb: 39`, `sv: 0`, `en: 36`, `fastai: 41`.
- Generates one single-core SLURM array per model in `model_evaluation/jobs/`; each task runs `model_evaluation/model_evaluation.py` on one project.
- `model_evaluation/job_error*.csv`, `model_evaluation/retry_en_timelimit.csv` and the `slurm_model_evaluation_retry_*.sh` scripts record and re-run the tasks that failed or exceeded the time limit.

## [`model_evaluation.ipynb`](model_evaluation.ipynb)

**Objective:** Compare the six retained models.

- Loads the final results from `model_evaluation/results/` for the retained configurations.
- Compares accuracy over the 360 projects with rank summaries, the Friedman test and the Conover post-hoc test.
- Compares accuracy and one-core computational time, including the Pareto frontier.
- Repeats the comparison for each PSPLIB set (`J30`, `J60`, `J90`, `J120`).
- Computes bootstrap confidence intervals, paired effect sizes and pairwise win rates.
- Saves the analysis tables in `model_evaluation/analysis/`.

## [`topology_analysis.ipynb`](topology_analysis.ipynb)

**Objective:** Relate model performance to project topology.

- Joins the evaluation results with the topological descriptors of `PSPLIB_sample/psplib_sample_metadata.csv`.
- Describes the distributions and correlations of the descriptors.
- Relates error and rank to the serial/parallel indicator `SP`.
- Reports Spearman correlations between each descriptor and model performance.
- Builds a statistical comparison grid over `SP` × `TF` quartiles.
- Saves the derived tables in `model_evaluation/analysis/`.

## [`model_evaluation/ols_benchmark_extension.py`](model_evaluation/ols_benchmark_extension.py)

**Objective:** Add ordinary least squares (OLS) as an untuned linear reference.

- `fit`: fits OLS on each evaluation project with the same scenarios, split and RelMSE definition as `model_evaluation.py`.
- `analyse`: repeats the Friedman test, the Conover post-hoc test and the summary statistics with seven models (six families + OLS), as an extension of the six-family analysis, whose tables are not modified.
- Writes to the folder given by `--out` (default `ols_out/`); the resulting tables are stored in `model_evaluation/analysis/` as `ols_relmse_360.csv` and `seven_model_*.csv`.
- Usage: `python model_evaluation/ols_benchmark_extension.py fit --out ols_out`, then `python model_evaluation/ols_benchmark_extension.py analyse --out ols_out`.
