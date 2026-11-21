# Hyperparameters

This document summarizes the hyperparameters explored during model selection. The configurations were generated in [`latinhypercube.ipynb`](latinhypercube.ipynb) using Latin Hypercube Sampling (LHS), with 50 sampled configurations per model (`Nlhs = 50`).

Each hyperparameter is sampled independently over a fixed range and the sampled columns are randomly permuted to create the final LHS combinations. Linear ranges are sampled using evenly spaced midpoints, while log-uniform ranges are sampled using evenly spaced midpoints in logarithmic space. Because midpoint sampling is used, the minimum and maximum values found in the generated CSV files are slightly inside the nominal ranges listed below.

The generated parameter files are stored in [`model_selection/hyperparameters/`](model_selection/hyperparameters/).

## Random Forest

Random Forest is an ensemble bagging model based on multiple decision trees. The selected hyperparameters control the number of trees, the complexity of each tree, and the randomness introduced at each split.

Parameter | Type | Sampling scale | Nominal range | Description
--- | --- | --- | --- | ---
`n_estimators` | Integer | Linear | 100 to 800 | Number of trees in the forest. Larger values can improve stability but increase computational cost.
`max_depth` | Integer | Linear | 3 to 30 | Maximum depth of each tree. This controls tree complexity and helps limit overfitting.
`min_samples_leaf` | Integer | Log-uniform | 1 to 50 | Minimum number of samples required at a leaf node. Larger values regularize the model.
`max_features` | Float | Linear | 0.3 to 1.0 | Fraction of features considered at each split. Lower values increase diversity among trees.

Generated file: [`lhs_rf_parameters.csv`](model_selection/hyperparameters/lhs_rf_parameters.csv)

## XGBoost

XGBoost is a gradient boosting model based on sequentially fitted decision trees. The selected hyperparameters control boosting complexity, learning dynamics, row and column subsampling, and regularization.

Parameter | Type | Sampling scale | Nominal range | Description
--- | --- | --- | --- | ---
`n_estimators` | Integer | Linear | 100 to 1000 | Number of boosting trees. It controls model capacity and computational cost.
`learning_rate` | Float | Log-uniform | 0.01 to 0.3 | Step size shrinkage used at each boosting iteration. Smaller values usually require more trees.
`max_depth` | Integer | Linear | 2 to 10 | Maximum depth of each tree. This controls the complexity of individual trees.
`subsample` | Float | Linear | 0.5 to 1.0 | Fraction of training observations sampled for each boosting round.
`colsample_bytree` | Float | Linear | 0.5 to 1.0 | Fraction of features sampled for each tree.
`min_child_weight` | Float | Log-uniform | 1 to 20 | Minimum sum of instance weight needed in a child node. Larger values make the model more conservative.
`reg_alpha` | Float | Log-uniform | 0.01 to 1.0 | L1 regularization term on weights.
`reg_lambda` | Float | Log-uniform | 0.01 to 10.0 | L2 regularization term on weights.

`gamma` is not sampled in the current workflow and is left at its default value.

Generated file: [`lhs_xgb_parameters.csv`](model_selection/hyperparameters/lhs_xgb_parameters.csv)

## Explainable Boosting Machine

Explainable Boosting Machine (EBM) is an interpretable generalized additive model trained through boosting. The selected hyperparameters control the amount of interaction modeling, boosting speed, feature-function complexity, and early stopping.

Parameter | Type | Sampling scale | Nominal range | Description
--- | --- | --- | --- | ---
`interactions` | Float | Linear | 0 to 0.9 | Fraction or amount of pairwise feature interactions included in the model.
`learning_rate` | Float | Log-uniform | 0.005 to 0.2 | Learning rate for the boosting process.
`max_leaves` | Integer | Linear | 2 to 4 | Maximum number of leaves used in the learned feature functions.
`min_samples_leaf` | Integer | Log-uniform | 1 to 100 | Minimum number of samples required at a leaf. Larger values regularize the learned functions.
`early_stopping_rounds` | Integer | Linear | 50 to 100 | Number of rounds used for early stopping to reduce overfitting.

Generated file: [`lhs_ebm_parameters.csv`](model_selection/hyperparameters/lhs_ebm_parameters.csv)

## Support Vector Regression

Support Vector Regression (SVR) is a kernel-based regression model. The current workflow assumes the default RBF kernel configuration and samples the regularization strength and epsilon-insensitive loss width.

Parameter | Type | Sampling scale | Nominal range | Description
--- | --- | --- | --- | ---
`C` | Float | Log-uniform | 0.1 to 100 | Regularization parameter. Larger values penalize errors more strongly and can produce less regularized models.
`epsilon` | Float | Linear | 0.01 to 0.5 | Width of the epsilon-insensitive tube in the SVR loss function.

The notebook leaves `gamma` at the scikit-learn default value (`scale`) and uses the default RBF kernel.

Generated file: [`lhs_sv_parameters.csv`](model_selection/hyperparameters/lhs_sv_parameters.csv)

## Elastic Net

Elastic Net is a regularized linear regression model combining L1 and L2 penalties. In this project, it is used with interaction terms generated through polynomial features.

Parameter | Type | Sampling scale | Nominal range | Description
--- | --- | --- | --- | ---
`alpha` | Float | Log-uniform | 0.0001 to 1.0 | Overall strength of regularization.
`l1_ratio` | Float | Linear | 0 to 1 | Balance between L1 and L2 regularization. Values near 0 are closer to Ridge regression; values near 1 are closer to Lasso regression.

Generated file: [`lhs_en_parameters.csv`](model_selection/hyperparameters/lhs_en_parameters.csv)

## FastAI Tabular Neural Network

The FastAI model is a tabular neural network. The sampled hyperparameters control the maximum learning rate, the width of two hidden layers, and weight decay.

Parameter | Type | Sampling scale | Nominal range | Description
--- | --- | --- | --- | ---
`lr_max` | Float | Log-uniform | 0.00001 to 0.1 | Maximum learning rate used during training.
`layer1` | Integer | Linear | 100 to 500 | Number of neurons in the first hidden layer.
`layer2` | Integer | Linear | 50 to 250 | Number of neurons in the second hidden layer.
`wd` | Float | Log-uniform | 0.0001 to 0.01 | Weight decay regularization.

Training settings that are not sampled: two hidden layers whose widths are rounded down to multiples of 5, batch size 64, the one-cycle learning-rate policy (`fit_one_cycle`) for at most 100 epochs, early stopping on the validation loss with patience 10, and a validation split of 10% of the training scenarios. Activity durations that are constant in the training data (activities completed at the control point) are removed before training, because fastai's normalisation maps them to non-zero test values.

Generated file: [`lhs_fastai_parameters.csv`](model_selection/hyperparameters/lhs_fastai_parameters.csv)

## Retained configurations

The configurations retained in [`model_selection.ipynb`](model_selection.ipynb) and used in the final evaluation are EBM 24, Elastic Net 36, Random Forest 14, XGBoost 39, SVR 0 and FastAI 41 (row indices of the LHS files, starting at 0). Their values are listed in [`model_selection/selected_models_results.csv`](model_selection/selected_models_results.csv).

## Generated LHS Files

Model | File | Number of configurations
--- | --- | ---
Random Forest | [`lhs_rf_parameters.csv`](model_selection/hyperparameters/lhs_rf_parameters.csv) | 50
XGBoost | [`lhs_xgb_parameters.csv`](model_selection/hyperparameters/lhs_xgb_parameters.csv) | 50
Explainable Boosting Machine | [`lhs_ebm_parameters.csv`](model_selection/hyperparameters/lhs_ebm_parameters.csv) | 50
Support Vector Regression | [`lhs_sv_parameters.csv`](model_selection/hyperparameters/lhs_sv_parameters.csv) | 50
Elastic Net | [`lhs_en_parameters.csv`](model_selection/hyperparameters/lhs_en_parameters.csv) | 50
FastAI tabular neural network | [`lhs_fastai_parameters.csv`](model_selection/hyperparameters/lhs_fastai_parameters.csv) | 50

## References

- [Random Forest Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- [XGBoost parameters](https://xgboost.readthedocs.io/en/stable/parameter.html)
- [Explainable Boosting Regressor](https://interpret.ml/docs/python/api/ExplainableBoostingRegressor.html)
- [Support Vector Regression](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
- [Elastic Net](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)
- [FastAI tabular models](https://docs.fast.ai/)
