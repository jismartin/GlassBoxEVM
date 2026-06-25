# GlassBox EVM

GlassBox EVM is a benchmarking repository for stochastic Earned Value Management (EVM). The project evaluates how different regression models predict project outcomes from Monte Carlo simulations of PSPLIB project schedules, with a specific focus on comparing the Explainable Boosting Machine (EBM) against other machine learning approaches.

The repository combines:

- PSPLIB project instances and topological metadata.
- Monte Carlo simulation outputs generated from stochastic project durations.
- Hyperparameter selection experiments based on Latin Hypercube Sampling.
- Final model evaluation results across held-out projects.
- Statistical and topology-aware analyses of model performance.

## Project Goal

The main goal is to benchmark interpretable and non-interpretable regression models for stochastic EVM forecasting. The workflow uses a fixed sample of PSPLIB projects, simulates stochastic project progress from an EVM control point, trains/evaluates models on Monte Carlo replications, and compares predictive accuracy across model families and project topologies.

## Benchmarked Models

The project evaluates the following regression models:

- [Explainable Boosting Machine (EBM)](https://interpret.ml/docs/python/api/ExplainableBoostingRegressor.html): interpretable GAM ensemble boosting model.
- [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html): ensemble bagging model.
- [Extreme Gradient Boosting (XGBoost)](https://xgboost.readthedocs.io/en/stable/python/python_api.html): ensemble boosting model.
- [Support Vector Regression (SVR)](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html): kernel-based regression model.
- [Elastic Net](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html): regularized linear model.
- [FastAI tabular neural network](https://docs.fast.ai/): deep learning model for tabular data.

## Repository Structure

- [`PSPLIB_sample/`](PSPLIB_sample/): selected PSPLIB instances, combined metadata, topology metadata, and topology validation notebook.
- [`montecarlo/`](montecarlo/): Monte Carlo simulation CSV files and fixed test-run indexes.
- [`model_selection/`](model_selection/): hyperparameter grids, selection scripts, SLURM jobs, and selection results.
- [`model_evaluation/`](model_evaluation/): final evaluation scripts, jobs, results, and analysis tables.
- [`scripts/`](scripts/): shared project utilities for reading PSPLIB files, computing topology, evaluating EVM control points, and running simulations.
- [`workflow_notebooks.md`](workflow_notebooks.md): ordered description of the active notebooks and the actions performed in each one.
- [`hyperparameters.md`](hyperparameters.md): description of the hyperparameters explored for each benchmarked model and the LHS ranges used during model selection.

## Workflow Overview

At a high level, the active workflow is:

1. Validate the PSPLIB sample and topological metadata.
2. Generate or verify Monte Carlo simulations from stochastic project durations.
3. Create project-level splits for hyperparameter selection and final evaluation.
4. Generate Latin Hypercube hyperparameter configurations.
5. Run model selection on the selection projects.
6. Choose the final LHS configuration for each model family.
7. Run final evaluation on the held-out projects.
8. Analyze global performance, subset-level performance, statistical significance, and topology-dependent behavior.

The detailed notebook order is documented in [`workflow_notebooks.md`](workflow_notebooks.md).

## Data Splits

The project uses a PSPLIB sample organized by subset (`J30`, `J60`, `J90`, `J120`). The active workflow separates projects into:

- Model selection projects: used to compare LHS hyperparameter configurations.
- Model evaluation projects: held out for the final benchmark.
- Monte Carlo run split: a fixed 80/20 train-test split over simulation replications, stored in [`montecarlo/test_runs.csv`](montecarlo/test_runs.csv).

## Environment

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

The main dependencies include `pandas`, `numpy`, `scikit-learn`, `xgboost`, `interpret`, `fastai`, `torch`, `networkx`, `scipy`, `scikit-posthocs`, `matplotlib`, and `seaborn`.

## Reproducibility Notes

- The notebook workflow assumes the repository is executed from the project root.
- Monte Carlo simulations and model runs can be computationally expensive.
- SLURM notebooks generate job scripts configured for the Altamira Supercomputer, especially for large-scale hyperparameter selection and final evaluation.

## Acknowledgements

We acknowledge the Santander Supercomputación support group at the University of Cantabria for providing access to the Altamira Supercomputer at the Institute of Physics of Cantabria (IFCA-CSIC), a member of the Spanish Supercomputing Network, which was used to perform the simulations and analyses.
