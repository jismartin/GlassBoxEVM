# GlassBoxEVM

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24425847.svg)](https://doi.org/10.5281/zenodo.24425847)
[![Data DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24439968.svg)](https://doi.org/10.5281/zenodo.24439968)

GlassBoxEVM is a benchmark of interpretable and black-box regression models used as surrogates of stochastic project duration. For each PSPLIB project, the stochastic activity durations from an Earned Value Management (EVM) control point are simulated by Monte Carlo, and one model per project is trained to map the simulated activity durations to the project makespan. The benchmark compares the accuracy and the computational cost of six model families, with particular attention to the interpretable Explainable Boosting Machine (EBM) and Elastic Net (EN).

The models are surrogates of a known simulation function (the makespan of a precedence network without resource constraints); they are not forecasts validated on real projects.

A second part (Sections 6–7 of the paper) asks when the interpretable models lose accuracy against the neural network, using the competition between critical sets as a structural explanation, and whether their explanations are faithful and useful for choosing interventions. It was tested on 320 new PSPLIB networks under a frozen protocol ([`protocols/sections_6_7.md`](protocols/sections_6_7.md)); the results are reported in [`protocols/confirmatory_results.md`](protocols/confirmatory_results.md).

The repository contains:

- the PSPLIB project instances used and their topological descriptors;
- the Monte Carlo simulation outputs;
- the hyperparameter search (Latin Hypercube Sampling) and its results;
- the final evaluation results on held-out projects;
- the statistical and topology-related analyses of model performance;
- the Sections 6–7 protocol, pipeline code, network manifests, confirmatory aggregates and the paper tables and figures.

## Benchmarked models

- [Explainable Boosting Machine (EBM)](https://interpret.ml/docs/python/api/ExplainableBoostingRegressor.html): interpretable generalised additive model with pairwise interactions, trained by boosting.
- [Elastic Net (EN)](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html): regularised linear model on the activity durations and their pairwise products.
- [Random Forest (RF)](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html): bagging ensemble of regression trees.
- [XGBoost (XGB)](https://xgboost.readthedocs.io/en/stable/python/python_api.html): gradient-boosted regression trees.
- [Support Vector Regression (SVR)](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html): kernel regression with an RBF kernel.
- [fastai tabular neural network](https://docs.fast.ai/): multilayer perceptron for tabular data. Activity durations that are constant in the training data (activities completed at the control point) are removed before training.

Ordinary least squares (OLS) on the standardised non-constant durations is added as an untuned linear reference in a separate seven-model analysis (`model_evaluation/ols_benchmark_extension.py`).

The search space of each family is described in [`hyperparameters.md`](hyperparameters.md).

## Data and experimental design

**Projects.** The project networks come from PSPLIB (Kolisch and Sprecher, 1997). The sample contains 400 single-mode instances, 100 from each of the J30, J60, J90 and J120 sets, stored in [`PSPLIB_sample/`](PSPLIB_sample/) together with their topological descriptors (Vanhoucke et al., 2008). In each set, 10 projects (40 in total) are used for hyperparameter selection and the remaining 90 (360 in total) for the final evaluation ([`model_selection/project_selection.csv`](model_selection/project_selection.csv), [`model_evaluation/project_evaluation.csv`](model_evaluation/project_evaluation.csv)). Resource constraints are not used.

**Simulation.** Activity durations follow independent beta-PERT distributions with minimum 0.8d, mode 1.1d and maximum 1.2d of the nominal duration d, and shape parameter λ = 4. The project state is evaluated at the control point EV = 15% of the budget at completion; completed activities keep their actual durations and in-progress activities are simulated for their remaining part. Each project has 10,000 simulated scenarios, stored in [`montecarlo/`](montecarlo/) as `<project>_simulation_01.csv` (columns `duration<n>` for activity n and `actual_duration` for the makespan).

**Train/test split.** The same 2,000 scenario indices (20%) are used as the test set for every project and model ([`montecarlo/test_runs.csv`](montecarlo/test_runs.csv)); the remaining 8,000 scenarios are used for training.

**Error measure.** RelMSE is the test mean squared error of a model divided by the test mean squared error of a constant predictor equal to the training mean. Values below 1 indicate improvement over the mean predictor.

**Hyperparameter selection.** Fifty LHS configurations per family were evaluated on the 40 selection projects. The configuration with the best mean rank across projects was retained, with rank variability as a secondary criterion: EBM 24, EN 36, RF 14, XGB 39, SVR 0 and fastai 41 ([`model_selection/selected_models_results.csv`](model_selection/selected_models_results.csv)).

**Final evaluation.** Each retained configuration was fitted and evaluated on each of the 360 evaluation projects as a single-core SLURM array task on the Altamira supercomputer. The reported times are wall-clock fit-and-predict times on one core.

## Repository structure

- [`PSPLIB_sample/`](PSPLIB_sample/): the 400 PSPLIB instances (`.sm`), sample and topology metadata, and the topology validation notebook.
- [`montecarlo/`](montecarlo/): simulated scenarios for the 400 projects and the fixed test indices.
- [`model_selection/`](model_selection/): LHS hyperparameter files, the selection script, SLURM job scripts and per-project selection results.
- [`model_evaluation/`](model_evaluation/): the evaluation script, SLURM job scripts and their error/retry logs, per-project results and analysis tables.
- [`scripts/tools.py`](scripts/tools.py): utilities to read PSPLIB files, compute topological indicators, evaluate the EVM control point and run the Monte Carlo simulation.
- [`results67/`](results67/): Sections 6–7 confirmatory aggregates (320 networks) and the pilot outputs used for Figure 6; the confirmatory report is [`protocols/confirmatory_results.md`](protocols/confirmatory_results.md).
- [`paper_out/`](paper_out/): paper Tables 4–8 and Figures 2–7, written by [`scripts/paper_outputs.py`](scripts/paper_outputs.py).
- [`workflow_notebooks.md`](workflow_notebooks.md): the notebooks and scripts in execution order and what each one does.
- [`hyperparameters.md`](hyperparameters.md): hyperparameters explored for each model family.

### Result files

- `model_selection/results/lhs_model_selection_<model>_project_<project>.csv`: RelMSE and time of the 50 LHS configurations on one selection project.
- `model_evaluation/results/model_evaluation_<model>_lhs_<id>_subset_<set>_project_<project>.csv`: hyperparameters, RelMSE and time of the retained configuration on one evaluation project.
- `model_evaluation/analysis/`: summary tables written by `model_evaluation.ipynb`, `topology_analysis.ipynb` and `model_evaluation/ols_benchmark_extension.py`: rank summaries, Friedman and Conover tests, bootstrap intervals, paired effect sizes, win rates, accuracy–time trade-offs, topology correlations, and the seven-model analysis with OLS (`ols_relmse_360.csv` and `seven_model_*.csv`).

## Workflow

1. Validate the PSPLIB sample and its topological metadata.
2. Simulate the Monte Carlo scenarios from the control point and define the project and scenario splits.
3. Generate the LHS hyperparameter configurations.
4. Run model selection on the 40 selection projects and retain one configuration per family.
5. Run the final evaluation on the 360 evaluation projects.
6. Analyse global and subset-level performance, statistical significance, uncertainty, cost and topology-related behaviour.
7. Fit the OLS reference and compute the seven-model extension.

The notebooks and scripts for each step are listed in [`workflow_notebooks.md`](workflow_notebooks.md).

## Environment

All model runs used Python 3.9 with NumPy 2.0.2, pandas 2.3.2, SciPy 1.13.1, scikit-learn 1.6.1, xgboost 2.1.4, interpret 0.7.2, PyTorch 2.8.0 (CPU build) and fastai 2.7.19. The pinned versions are listed in [`requirements.txt`](requirements.txt). fastai 2.7.19 declares `torch<2.7`, so it is installed without dependency resolution:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install --no-deps fastai==2.7.19
```

The analysis notebooks additionally use seaborn and scikit-posthocs, and the seven-model extension uses mpmath.

## Reproducibility notes

- Notebooks and scripts are run from the repository root (except `PSPLIB_sample/psplib_sample_topology.ipynb`, which runs from its own folder).
- Model runs use the seed 112358. RelMSE values of the fastai models can differ between machines; the reported fastai results are those obtained on Altamira.
- The SLURM scripts are configured for the Altamira supercomputer; the cluster paths in them must be adapted for other systems.
- The simulation outputs and all per-project results are included, so every analysis table can be recomputed without re-running the simulations or the model fits.

## Licence

The code is released under the MIT licence (see [`LICENSE`](LICENSE)). The simulation outputs and result tables are released under the Creative Commons Attribution 4.0 International licence (CC BY 4.0). The PSPLIB instances in `PSPLIB_sample/` are redistributed from PSPLIB; please cite Kolisch and Sprecher (1997) when using them.

## Citation

Please cite this repository using the metadata in [`CITATION.cff`](CITATION.cff). The benchmark release `v1.0-benchmark` is archived on Zenodo: [https://doi.org/10.5281/zenodo.24425847](https://doi.org/10.5281/zenodo.24425847). The complete cluster outputs of the Sections 6–7 study (fitted models, predictions, simulated scenarios, explanations and decisions, with a SHA-256 manifest) are archived as a separate dataset: [https://doi.org/10.5281/zenodo.24439968](https://doi.org/10.5281/zenodo.24439968).

## References

- Kolisch, R., and Sprecher, A. (1997). PSPLIB – A project scheduling problem library. *European Journal of Operational Research*, 96(1), 205–216.
- Vanhoucke, M., Coelho, J., Debels, D., Maenhout, B., and Tavares, L. V. (2008). An evaluation of the adequacy of project network generators with systematically sampled networks. *European Journal of Operational Research*, 187(2), 511–524.

## Acknowledgements

We acknowledge the Santander Supercomputación support group at the University of Cantabria for providing access to the Altamira Supercomputer at the Institute of Physics of Cantabria (IFCA-CSIC), a member of the Spanish Supercomputing Network, which was used to perform the simulations and analyses.
