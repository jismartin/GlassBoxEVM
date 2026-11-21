# Analysis protocol for the structural and explanation analyses (Sections 6–7)

**Project:** GlassBoxEVM — interpretable and black-box surrogates of stochastic project duration.
**Protocol version:** 1.1 (frozen).
**Date:** 21 November 2026.
**Status:** frozen after the technical pilot and before any confirmatory network was simulated. The frozen version is identified by the Git tag `sections67-protocol-v1.1` (§10).

## 1. Purpose and scope

The study compares the accuracy and construction cost of six regression families as surrogates of stochastic project duration on PSPLIB networks, with particular attention to two interpretable models, Elastic Net (EN) and the Explainable Boosting Machine (EBM). This protocol specifies the analyses reported in Sections 6 and 7 of the article. Those analyses explain the differences observed in the benchmark and assess whether the interpretable models reproduce the effects of activity durations and support a simulated intervention decision. They are not intended as an exhaustive characterisation of network complexity or as a catalogue of sensitivity indices.

For each network, a separate surrogate is trained to approximate the function g that maps simulated activity durations to project makespan. The models are surrogates of a known simulation function; the study does not validate forecasts of real projects from information available during execution. Without resource constraints, g is the maximum over paths of the sum of activity durations, a piecewise-linear function. Evaluating g directly provides the reference for accuracy, effect fidelity and intervention outcomes.

| Research question | Evidence |
| --- | --- |
| RQ1. How accurate are the six families, and where do EN and EBM stand? | Section 5: benchmark on 360 networks, with OLS as an additional linear reference. |
| RQ2. Under which conditions do the differences between EN, EBM and fastai change? | Section 6: critical-set competition, departure from linearity, size and ablations. |
| RQ3. What does each model cost to build? | Section 5: one-core fit-and-predict times and the accuracy–cost trade-off. |
| RQ4. Do the surrogates reproduce activity effects, and do they select useful interventions? | Section 7: local effects compared with g and a simulated decision test. |

The six-family benchmark remains the main comparison. The focused analyses use EN, EBM and fastai, the three families whose benchmark results raise the explanatory questions. No model is expected a priori to outperform the others, and C is not assumed to account for all error differences.

## 2. Networks, data and simulation

### 2.1 Network sets

- **Selection set:** the 40 networks used to select hyperparameters. They are excluded from Sections 6 and 7.
- **Exploratory set:** the 360 evaluation networks of the benchmark. They were used to formulate the hypotheses below, to choose illustrative examples and to run the technical pilot.
- **Confirmatory set:** 320 networks not previously used, 80 from each of J30, J60, J90 and J120. Each network receives its own per-network fits. The set tests patterns across networks; it does not test the transfer of a single model between networks.

The confirmatory networks are drawn by simple random sampling within each subset. The sampling frame is `PSPLIB_complete/psplib_complete.csv`; candidates are the rows with `sample == False` (380 in each of J30, J60 and J90; 500 in J120), which excludes all selection and evaluation networks. One generator is used per subset, `numpy.random.default_rng([20261119, k])` with k = 0, 1, 2, 3 for J30, J60, J90 and J120, and it permutes the candidates in the numeric order of the CSV (parameter combination, then instance). The first 80 networks of each permutation form the sample and the remaining ones, in order, form the substitute list. The full permutation is saved before any simulation.

A network may be excluded before simulation only if its file cannot be read, the control point cannot be computed, or the topological indicators cannot be computed. The reason is recorded and the next network in the permutation is taken. No network is excluded or replaced on the basis of its results.

Network identifiers and SHA-256 hashes of the PSPLIB files are stored. Exploratory and confirmatory results are reported separately; a pooled summary of the 680 networks, if shown, does not replace that separation.

### 2.2 Simulation

The simulator of the benchmark is used unchanged: independent beta-PERT activity durations with minimum 0.8d, mode 1.1d, maximum 1.2d and λ = 4; a control point at EV = 0.15 BAC; and 10,000 scenarios per network. Completed activities are fixed at their actual durations, and the remaining durations of in-progress activities are simulated as in the benchmark. The control-point state is computed deterministically from mean durations (`evaluate_at_EV`), and each in-progress activity is an affine transformation of an independent beta draw. The stochastic inputs are therefore independent by construction.

All models of a network share the same scenarios and the same training/test partition. `montecarlo/test_runs.csv` contains a single list of 2,000 scenario indices (20%), common to all networks, and is used for the confirmatory networks as well.

The benchmark scenarios were drawn from NumPy's global generator without a recorded seed, so they cannot be regenerated. They are stored in `montecarlo/` and are reused as stored for the exploratory networks. Scenarios for the confirmatory networks, and all additional scenario samples, are generated with explicit seeded generators (§9.2), with the same distributions and control point. The training seed of the benchmark (112358) is retained. All preprocessing is estimated on training data only.

### 2.3 Additional scenario samples

For each network analysed in Section 7, independent samples are generated for the decision test (5,000 selection and 5,000 evaluation scenarios) and for the Sobol indices (§6.2). Further simulations are used only for the numerical-precision checks that require them.

## 3. Models

### 3.1 Benchmark

The benchmark comprises EBM, EN, XGBoost, Random Forest, SVR and a fastai MLP, each with the configuration retained in the selection stage. It reports mean and median error, error tails, wins, ranks and times. OLS is added as a linear reference without altering the configuration selection. A Friedman test with post-hoc comparisons that includes OLS is computed as a separate seven-model analysis and labelled as an extension of the six-family analysis. The benchmark is not repeated on the confirmatory networks.

### 3.2 Focal models and ablations

| Model | Definition and purpose |
| --- | --- |
| EN-36 | Retained configuration: pairwise products of distinct durations (uncentred), followed by standardisation and Elastic Net; alpha 1.318e-4, l1_ratio 0.91, max_iter 50,000. |
| EBM-24 | Retained configuration, including interactions 0.873, learning_rate 0.01569, max_leaves 3, min_samples_leaf 3 and early_stopping_rounds 74. |
| fastai-41 | Retained configuration; constant inputs are removed before training. Flexible reference. |
| OLS | Linear regression with intercept on standardised non-constant durations, without regularisation. |
| EBM-24-noint | EBM-24 with interactions = 0; isolates the contribution of pairwise terms. |
| EN-36-main | EN-36 without product terms; isolates the contribution of those terms. |
| EN-36-centred | EN-36 with durations centred at their training means before forming products; isolates the effect of parametrisation under penalisation. |

Complete configurations, including options not listed here, library versions and seeds, are exported from the benchmark code. Ablations are not retuned; their results describe the effect of removing a component from the retained configuration, not the best variant attainable by a new search. Ablation results are reported mainly for the confirmatory networks, with pilot results identified separately.

### 3.3 Fitting and persistence

The benchmark stored errors and times but not fitted models. Every analysis that needs a fitted model therefore uses a refit with the same configuration, scenarios, partition and seed. This applies to the pilot networks, the illustrative networks (§8) and the stability networks (§6.2).

Each fit stores the model, preprocessing, column order, constant-input mask, partition indices and metadata. Predictions are checked to be reproduced after reloading. Fitting and post-fit analyses are separate stages. Benchmark timing comparisons use one core per task; the cost of post-fit analyses is reported separately.

## 4. Structural measures (Section 6)

### 4.1 Core measures

**Critical set.** In scenario r, an activity is critical if its total float is below 1e-9 · T_r, where T_r is the makespan. Ties are included, so the critical set contains all critical activities. Sets are compared for identity, not partial similarity.

**Competition C = 1 − relative frequency of the most frequent critical set** over the 10,000 scenarios. C measures how often the critical set departs from the dominant configuration, not the magnitude of the consequences. C = 0 is described as no observed change of critical set. Sensitivity to tolerances of 1e-7 and 1e-5 is reported; 1e-3 corresponds to near-criticality and is not used.

**Departure from linearity.** Out-of-sample RelMSE of OLS: the sum of squared OLS test errors divided by the sum of squared test errors of the training-mean predictor. It describes the loss from approximating g linearly; it is not identified with a population quantity or with the pure cost of critical-set changes.

**Size and topology.** Subset, number of activities, serial/parallel indicator SP and topological float TF. SP and TF describe structure only, whereas C also depends on duration distributions and on the project state at the control point. Associations with topology are reported descriptively; no superiority of C over SP is assumed.

Networks with C = 0 are shown in a separate panel and are retained in all global summaries. Small rank differences near zero error are not interpreted as practically important.

### 4.2 Supporting decomposition

When it passes numerical validation, the population decomposition normalised by Var(g),

`L* = Q + I_int`,

is used to support the interpretation. L* is the error of the best population linear predictor, Q the curvature of the single-activity effects that the linear predictor cannot represent, and I_int the residual of the best additive approximation, which arises from interactions. The identity requires independent inputs.

An empirical estimator is used. On the 10,000 scenarios, g is regressed by least squares on the non-constant durations and L_emp = mean(linear residual²)/Var_emp(g) (ddof = 0). Partial-dependence curves are estimated and centred to give an additive approximation g_add; then I_int_emp = mean((g − g_add)²)/Var_emp(g) and Q_emp = L_emp − I_int_emp. Output files keep these names distinct from the population quantities. Q_emp may be slightly negative because of estimation error; it is neither truncated nor interpreted as negative curvature. I_int is estimated as a residual, not as 1 − ΣS1. The out-of-sample OLS RelMSE remains a separate diagnostic, because its finite-sample fit and denominator differ from L*.

On five pilot networks, the additive approximation was evaluated on an independent simulation and Q_emp was compared with a direct estimate of single-activity curvature. A 41-point grid changed Q_emp by at most 1.3e-5 and the direct curvature estimate agreed with Q_emp within 2.2e-4, whereas evaluation on an independent sample changed L_emp, I_int_emp and Q_emp by up to 0.004 of Var(g). This sampling resolution, ±0.004 of Var(g), is declared for the decomposition, and smaller differences are not interpreted. This check does not condition H1–H4.

The ratio Q_emp/L_emp is reported only when L_emp ≥ 1e-3 and the numerical resolution allows interpretation. Its association with the EN−EBM difference is analysed descriptively and is not a confirmatory hypothesis.

Secondary descriptors are N_eff = exp(−Σ p_s log p_s), the effective number of critical sets; the criticality index CI of each activity; and the proportion of non-constant activities with 0.05 < CI < 0.95.

### 4.3 Exploratory analyses

The split of I_int into pure pairwise and higher-order components is reported in an appendix as an exploratory analysis of the 360 exploratory networks. The report states the activity filter, the omitted pairs and the uncertainty of the residual; a residual that includes omitted pairs is not described as pure higher-order interaction.

Theoretical approximation bounds for each function class are used as internal consistency checks under independence and a common normalisation. They are not treated as hypotheses or as exact bounds on finite-sample test error. EN-36-main and OLS are not treated as equivalent, because EN-36-main is regularised.

Variation of duration ranges, set-similarity measures, additional model families and a full characterisation of interactions are outside the scope of this protocol. Any such extension is labelled exploratory.

## 5. Statistical analysis plan

### 5.1 Conventions

The unit of analysis is the network. For models a and b, Delta_a−b = RelMSE_a − RelMSE_b; a positive value favours b. All results of a network are resampled together.

Uncertainty is estimated with 10,000 bootstrap replicates of networks, stratified by subset, recomputing every statistic in each replicate. Descriptive intervals are 95% percentile intervals. They describe between-network variability conditional on the fitted models, not the full variability of training.

Mean differences weight each subset equally (one quarter) and each valid network equally within its subset. Regression models are fitted by least squares with subset-specific intercepts. If networks are lost unevenly across subsets, weights restore an equal total contribution from each subset. The number of valid pairs is always reported.

### 5.2 Primary hypothesis H1

For m ∈ {EN, EBM}:

`Delta_m−fastai,js = alpha_s + beta_m C_js + error_js`,

with four subset intercepts and one common slope.

H1 states that the advantage of fastai over each interpretable model increases, or its disadvantage decreases, as C increases. **H1 is confirmed only if the lower bounds of the two-sided 95% bootstrap intervals for both beta_EN and beta_EBM are greater than zero.** For each slope this corresponds approximately to a one-sided test at the 2.5% level. Because the claim requires both slopes to be positive, it is an intersection–union test and needs no multiplicity correction between the two slopes. H1 is the primary hypothesis, tested at its own level and outside the Holm family of §5.3.

Subset-specific slopes and LOESS curves are shown to describe heterogeneity. A positive slope does not imply that fastai is more accurate in absolute terms. Pointwise bands are not presented as simultaneous guarantees over a range of C.

### 5.3 Secondary hypotheses H2–H4

| Hypothesis | Confirmatory statistic (two-sided) |
| --- | --- |
| H2 | Difference beta_J30 − beta_J120 in Delta_EN−EBM = alpha_s + beta_s C + error (four intercepts, four slopes). |
| H3 | Paired mean EN−EBM of the primary local-effect error (§6.1). |
| H4 | Paired mean EN−EBM of the decision regret at k = 3 (§6.3). |

The three p-values are adjusted with Holm's procedure at a family-wise level of 0.05. This family does not control error across all analyses of the article. P-values are obtained from the null-centred bootstrap: the proportion of replicates with |theta_boot − theta_obs| ≥ |theta_obs|, with one added to numerator and denominator. This is a bootstrap approximation, not an exact test; its implementation is verified on reference examples before the confirmatory analysis. If a hypothesis cannot be estimated, the family is kept, its p-value is set to 1 for the adjustment, and it is reported as not estimable. Estimates, marginal intervals and raw and adjusted p-values are reported; the marginal intervals are not simultaneous.

### 5.4 Prespecified sensitivity analysis for H2

The least-squares contrast is the confirmatory analysis. Two non-confirmatory sensitivity analyses are also reported:

1. Theil–Sen slopes per subset (`scipy.stats.theilslopes`, `method='separate'`) and their J30 − J120 contrast, with a percentile interval from the same stratified bootstrap of 10,000 replicates.
2. Influence diagnostics for the least-squares H2 model: Cook's distance for each network, flagging networks with D > 4/n, and the change in the contrast when each network is removed in turn.

No network is excluded because of its influence. If the least-squares and Theil–Sen contrasts have the same sign and the Theil–Sen contrast is at least half the least-squares contrast in magnitude, the pattern is described as robust. Otherwise, the report states that the mean pattern is concentrated in a subset of networks and identifies them.

### 5.5 Reporting rules

For EN−EBM, the win frequency, median and mean of the paired differences are always reported together. An exact tie is Delta = 0; no equivalence margin is used, and a non-significant difference is not interpreted as equivalence. Thresholds such as C > 0.1 are descriptive only. Associations other than H1–H4, and the ablation results, are reported as explanatory analyses. The 360 exploratory and 320 confirmatory networks are not pooled for confirmatory tests, and statistical models are not modified after the confirmatory data are examined.

### 5.6 Sample size

The confirmatory sample has 80 networks per subset. The size was set to confirm H1 and to estimate H2 with useful precision; it will not be changed after results are observed. Projections from the 360 exploratory networks, rescaled to 80 networks per subset, are indicative:

- H1: slopes of 0.071 (EN − fastai) and 0.078 (EBM − fastai), with standard errors of 0.006 and 0.010 and a bootstrap correlation of 0.91. Joint power under the §5.2 criterion is 1.00, 1.00 and 0.98 if the true effects are 100%, 70% and 50% of the exploratory estimates.
- H2: a contrast of 0.061 with a standard error of 0.014. Power at the most stringent Holm step (α/3) is 0.97, 0.74 and 0.41 for 100%, 70% and 50% of the exploratory effect.
- H3 and H4: no exploratory estimates are available at scale. The 24 pilot networks, chosen to span the range of C, provide only indicative estimates of their variability.

## 6. Explanation fidelity and decision value (Section 7)

### 6.1 Local effects (primary fidelity measure)

A sample of 500 test scenarios, common to all models, is drawn with a recorded seed. Constant inputs are excluded. Each activity is perturbed by −1, −0.5, +0.5 and +1 standard deviations, estimated on training data.

Perturbations are clipped to the theoretical support of the modifiable duration defined by the simulator, which accounts for in-progress activities; this support differs from the range observed in training. The effective increment delta_ef is recorded, and a perturbation is excluded if |delta_ef| is below 10% of the nominal increment.

The local effect is D_i = [f(x + delta_ef e_i) − f(x)] / delta_ef, computed for each model f and for g. The primary error is the mean of |D_model − D_exact|, averaged first over valid scenarios for each activity and perturbation, then over perturbations, then over activities, with equal weights at each level. Activities without valid perturbations are recorded. The validity mask depends only on the reference and the domain, never on model performance.

Secondary measures are: the error in the duration change in time units; the proportion of D < −0.05 or D > 1.05, and the mean excess of D outside [0, 1], max(−D, 0) + max(D − 1, 0), averaged as the primary error; results separated for numerically null exact changes (|Delta g| < 1e-9 T) and non-null changes; and the frequency of critical-set changes. For negative perturbations, the sign of the duration change is distinguished from the response per unit, which is non-negative for g; a negative duration change is not counted as a violation when D is positive. A sample targeted at rare critical-set changes, if used, is reported separately.

### 6.2 Global explanations (supporting measures)

Partial-dependence (PD) curves of g and of each model are compared on a grid of 21 marginal quantiles between 1% and 99%, with the 500 test scenarios as common background. Curves are centred under the marginal distribution of each activity, and discrepancies are weighted by that distribution rather than uniformly over grid points. Fidelity is summarised by the integrated squared difference between centred curves, normalised by Var(g), with the curvature component shown separately. Activities whose exact curve is constant are reported separately and are not assigned an R².

First-order Sobol indices S1 are estimated from the variance of the PD curve and total indices ST with the Jansen estimator using permuted columns. Both are evaluated on an independent sample of 10,000 scenarios per network shared by g and all models; S1 uses a 500-row background drawn from that sample. Each function's indices use that function's own variance. In the pilot, three permutation seeds, 21- and 41-point grids and 500- and 2,000-row backgrounds changed individual indices of g by at most 0.005 (95th percentile 0.001), which is the declared resolution of S1 and ST. PD curves use the 21-point grid with linear extrapolation beyond the extreme nodes; a 41-point grid changed the PD error by less than 15% and adding the support limits to the grid increased it, mainly for EBM, without changing the ordering of the models in any of the five pilot networks checked. Analytical references are computed for constructed linear functions, for which S1_i = ST_i = b_i² Var(X_i)/Var(g), and zero for activities without effect. Observed C = 0 is not by itself taken as proof of linearity. The numerical resolution is reported, and differences below it are not interpreted.

The number of model terms and pairwise terms is reported. Agreement with the schedule-risk indices CI, SI, CRI and SSI is complementary evidence: those indices are not equated with model-internal importances, and EN coefficients are not summed to form activity sensitivities. Rankings are compared through a common per-activity measure, the variance of the activity's PD curve, with a fixed tie-breaking rule and number of eligible activities; cases with constant indices or undefined correlations are reported separately. Refit stability on 30 exploratory networks, with two resamples per network, is a secondary diagnostic reported in an appendix.

### 6.3 Decision test

The intervention caps an activity at its median, x'_i = min(x_i, m_i), where m_i is estimated on the 5,000 selection scenarios. For in-progress activities the cap applies to the remaining duration. Only non-completed, non-constant activities are eligible. The intervention mitigates the upper tail of an activity's duration within its support; it is not an economically validated action.

For k = 1, 2, 3, a sequential greedy policy selects at each step the activity that maximises the estimated reduction in mean makespan, given the activities already selected. The same rule is applied with g (exact-greedy) and with EN, EBM and fastai. Activities are not repeated, and ties are broken by identifier. CI and SSI, computed on the selection scenarios, provide static reference rankings.

All selected sets are evaluated with g on the 5,000 evaluation scenarios, which are not used during selection. Reductions in the mean and in the 90th percentile are reported; only the mean drives selection. The H4 statistic uses the regret at k = 3, defined as the exact-greedy benefit minus the surrogate-greedy benefit in mean duration. A network with fewer than three eligible activities is excluded from H4 and recorded; no other k is substituted. Negative regret is possible. Networks with C = 0 are included in the primary H4 analysis and are also described separately: C = 0 only indicates that no change of critical set was observed in the simulated scenarios, whereas the intervention produces other combinations of durations, and the models may estimate benefits differently and select different activities.

The value of k and the total expected reduction in the capped durations are reported. No equality of cost between policies, and no economic efficiency, is claimed. For J30 and J60, the exhaustive optimum on the selection sample is reported as an additional reference if it passes the pilot checks of cost and correctness; it is identified as a sample optimum, not a population optimum, and its absence does not affect H4. Additional importance-based rankings, equal-budget interventions and duration-shift interventions are optional extensions.

## 7. Pilot, quality control and failures

### 7.1 Technical pilot

The pilot uses 24 exploratory networks, six per subset, chosen to span the range of C. Refits use the stored benchmark scenarios and the original partition, so that a comparison with the benchmark RelMSE isolates the effects of refitting and platform; the newly generated samples (decision, Sobol) are not part of that comparison. The illustrative networks of §8 are refitted if they are not among the 24.

The pilot verifies: model persistence and reload; reproduction of the benchmark errors on the computing cluster; seeds; memory and time requirements; signs of paired differences; exact local effects within [0, 1]; precision of the Sobol indices and of Q/I_int (21- versus 41-point grids, different background sizes); and the availability of valid interventions. Reproduction discrepancies are investigated and documented, and platform tolerances are fixed, before the confirmatory run. On the computing cluster, refits are expected to reproduce the benchmark exactly; any difference is reported per network, and RelMSE differences below 1e-5 in absolute value are not investigated further. Computing resources for the confirmatory run are sized from the pilot timings. The pilot and its results are reported in `protocols/pilot_report.md`.

If sufficient precision is not achieved for Q/I_int or the higher-order decomposition, those analyses are reported as exploratory with their limitations, or omitted. This does not affect C, OLS, H1–H4 or the decision test.

### 7.2 Acceptance checks

Before the confirmatory run, the following are verified: identifiers and partitions match; reloaded models reproduce predictions; benchmark errors are reproduced within the platform tolerance; g does not decrease when an activity is lengthened; exact D lies in [0, 1] within numerical tolerance; capping never increases g in any scenario; the exhaustive search is not worse than greedy on the selection sample for the same k; metrics are invariant to row order; signs of differences and regret are correct; the Holm implementation reproduces a known example; and, for a constructed linear function, S1 = ST matches the analytical values, with zero for irrelevant activities. C = 0 is not used to require I_int = 0. Discrepancies larger than the validated precision are investigated without removing networks.

A dependent stage is stopped if retained configurations are missing, training and test data overlap, the exact function does not reproduce the simulated makespan, or reloaded models do not reproduce predictions. The cause is documented and corrected before continuing.

### 7.3 Failures

A failed fit is retried once with the same design, and both attempts are logged; the best of several attempts is never selected. A network with a persistent failure is excluded only from comparisons that require the failed model. The number of failures by model, subset and C is reported, and conclusions are stated as conditional on successful fits. If failures prevent an interpretable confirmation, this is reported; networks are not replaced on the basis of performance.

## 8. Reporting in the manuscript

**Section 5** reports the six-family comparison with OLS as a reference: overall accuracy, tails, wins and cost.

**Section 6** reports (6.1) C, the OLS deviation and topological context; (6.2) paired differences and H1–H2; (6.3) ablations and, where reliable, the Q/I_int decomposition; and (6.4) implications within the studied domain. The higher-order decomposition is reported in an appendix.

**Section 7** reports (7.1) local fidelity; (7.2) PD curves and global sensitivity as support; and (7.3) the decision test. Decision usefulness is not inferred from ranking correlations alone, and ease of human interpretation is not inferred from model structure alone.

The illustrative figure uses three exploratory networks selected by rule, without reference to model performance: C = 0 with the median OLS deviation among such networks; exactly two critical sets with C closest to 0.5; and the maximum C. Ties are broken by alphanumeric identifier. If a category is empty, this is stated and the panel is omitted. Each panel shows g and all focal models.

Associations are not interpreted causally. The discussion states the limits of the setting: synthetic PSPLIB networks, independent beta-PERT durations with a narrow range, a single control point, no resource constraints and per-network training.

## 9. Implementation

### 9.1 Code

The analyses are implemented in the package `scripts/sections67/` of the project repository, with a single command-line entry point `python -m scripts.sections67 <stage>`. The simulator (`scripts/tools.py`), the benchmark scripts and the retained configurations are used without modification. The full configuration, seeds and stage parameters are defined in `scripts/sections67/config.py`, and each output records the configuration hash, code hash and commit. The automated acceptance checks of §7.2 are in `scripts/sections67/tests/`.

### 9.2 Seeds and samples

The confirmatory sample is drawn as described in §2.1; the NumPy version, the candidate list, the algorithm and the full permutation with substitutes are recorded.

All other random samples derive from the root seed 20261121. The seed for a given network, stage and replicate is obtained from the SHA-256 digest of the string `root|network_id|stage|replicate`, taking its first 8 bytes as a little-endian integer modulo 2**32. The stages are `simulate`, `local_test`, `pd_background`, `sobol_sample`, `sobol_perm`, `decision_select`, `decision_eval`, `independent_validation`, `stability` and `network_bootstrap`. The `simulate` stage generates the 10,000 training and test scenarios of confirmatory networks; exploratory networks keep their stored scenarios. Samples and perturbations are shared by all models, and the concrete indices are stored together with the seeds.

The 500 local-effect scenarios and the PD background are drawn without replacement from the test set, independently of each other. If a network has fewer than 500 test scenarios, all are used and n is recorded. Standard deviations are computed on training data with ddof = 0. Constant columns follow the rule of the benchmark pipeline, and the mask is stored. If Var(g) or the RelMSE denominator is zero, the result is recorded as undefined; no epsilon is added.

### 9.3 Stages

- `prepare`: network inventory, sample manifests, exclusion checks and configuration snapshot.
- `simulate`: scenario generation, control-point metadata and the theoretical support of each modifiable duration.
- `fit`: one task per network and model; stores the fitted pipeline and predictions.
- `structure`: exact function check, C, CI, N_eff, topology, OLS deviation and, when enabled, Q/I_int.
- `explain`: local effects, PD curves and Sobol indices; stability refits as separate tasks.
- `decide`: policy selection on the selection sample and exact evaluation on the evaluation sample.
- `resolution`: the numerical resolution checks of §7.1 (pilot only).
- `aggregate`: complete-pair checks, per-network metrics, bootstrap inference and Holm adjustment, with cohorts kept separate.

Tasks are resumable per network, model and stage. An existing output is reused only if the hashes of inputs, configuration and code match. Writes are atomic (temporary file and rename), and earlier outputs are never silently overwritten. Fitting tasks use one core, with BLAS and OpenMP threads limited to one. After transfer from the cluster and SHA-256 verification, the original outputs are retained until a durable archive of models, data and metadata has been confirmed. Large files are not stored in Git; the repository holds the code, configuration and manifests that identify the archive.

### 9.4 Metric definitions

- RelMSE = Σ(y_test − ŷ_test)² / Σ(y_test − mean(y_train))².
- C uses all 10,000 scenarios and sorted sets of activity identifiers, ties included, with a tolerance relative to each scenario's makespan. CI is the frequency of membership in the critical set; the intermediate-criticality proportion uses non-constant activities only.
- Local effects store both Delta_y and D = Delta_y/delta_ef. H3 uses the difference between the per-network EN and EBM means. The violation rate uses the same validity mask.
- PD: 21 marginal quantiles between 0.01 and 0.99 of the stored scenarios, with duplicate nodes removed. f is evaluated on the common test background with one column replaced. Values between nodes are interpolated linearly, and values beyond the extreme nodes are extrapolated linearly up to the support limits; this treatment is compared in the pilot with a grid that includes the support limits. Each curve is centred by its mean over the empirical marginal values. PD error per activity = mean((PD_model,centred(X_i) − PD_exact,centred(X_i))²)/Var_emp(g); the per-network mean weights activities equally, and the per-activity distribution is also reported. R² is computed only when the exact curve has positive variance.
- S1 = Var(PD_f(X_i))/Var(f(X)); ST = mean((f(X) − f(X_perm,i))²)/(2 Var(f(X))), where X is the independent Sobol sample, with three permutation seeds summarised by mean and dispersion. The source function and denominator are stored with each index.
- Decision: at each step, f is evaluated on the selection scenarios with the chosen activities and each remaining candidate capped at their medians, and the candidate with the largest mean reduction is chosen. B_mean = mean(g(X_eval)) − mean(g(X_eval, capped)); B_p90 = P90(g(X_eval)) − P90(g(X_eval, capped)), using linearly interpolated quantiles. Regret = B_mean(exact-greedy) − B_mean(surrogate-greedy). H4 uses the EN−EBM difference in regret, in duration units. A version divided by the baseline mean duration is reported descriptively.
- CI and SSI on the selection scenarios: SSI_i = CI_i · sd(X_i)/sd(g(X)), with ddof = 0; the first k eligible activities are selected, with ties broken by identifier. If CRI and SI are also computed, CRI = |Pearson(X_i, g)| and SI = mean(X_i · g / ((X_i + float_i) · mean(g))), using the modifiable duration and the makespan consistently; null denominators are excluded and recorded.
- Exhaustive search: all sets of exactly k eligible activities (k = 1, 2, 3) are enumerated, the set with the smallest mean(g) on the selection sample is chosen, and only that set is evaluated on the evaluation sample. The number of combinations and the computation time are recorded.

### 9.5 Inference details

Each contrast uses complete pairs and the stratified bootstrap of §5.1. H1 uses the networks with valid EN, EBM and fastai results; if networks are lost, the loss is reported and the pairwise sets are shown as a sensitivity analysis. H2 requires valid EN and EBM results; H3 requires both local metrics; and H4 requires both surrogate policies, exact-greedy and at least three eligible activities. H2–H4 statistics are collected in a single ordered list before applying Holm. Marginal intervals are not described as Holm-adjusted.

### 9.6 Output files

| File | Content |
| --- | --- |
| network_manifest.csv | cohort, subset, network_id, source_sha256, status, exclusion_reason, seeds |
| candidate_permutation.csv | subset, rank, network_id, selected, substitute_order |
| fit_manifest.csv | cohort, network_id, model_id, config_hash, fit_status, attempts, runtime, memory, model_path |
| predictions | cohort, network_id, model_id, scenario_id, split, y_true, y_pred |
| structure_metrics.csv | cohort, subset, network_id, C, CI summaries, OLS_RelMSE, SP, TF, n_variable, N_eff |
| decomposition.csv | network_id, L_emp, I_int_emp, Q_emp, Q_over_L, resolution, grid, background_n, validity_flag |
| local_metrics.csv | cohort, network_id, model_id, MAE_D, MAE_time, violation_rate, valid_n, clipped_n, excluded_n, switch_n |
| sensitivity | cohort, network_id, model_id, activity_id, PD_error, S1, ST, permutation_seed |
| decisions.csv | cohort, network_id, policy, k, selected_ids, B_mean, B_p90, expected_duration_reduction, regret |
| inference.csv | cohort, hypothesis, statistic, estimate, ci_low, ci_high, p_raw, p_holm, n_by_subset |
| run manifest | protocol_version, code_commit, config_hash, environment, start/end, stage_status, deviations |

Detail tables are stored as gzip-compressed CSV or Parquet. Scenario identifiers are preserved, and modified durations are stored separately from the original ones. Stored detail is sufficient to recompute all aggregates without refitting.

## 10. Protocol record

### 10.1 Version history

| Version | Date | Commit | Description |
| --- | --- | --- | --- |
| 1.0 | 21 November 2026 | ca7fbdc | Pre-pilot version. |
| 1.1 | 21 November 2026 | tag `sections67-protocol-v1.1` | Frozen version: pre-freeze amendment after the technical pilot (§10.3); pipeline code at commit 9ec86e3. |

The frozen version, with the technical parameters fixed in the pilot, is recorded here before any confirmatory network is analysed. Any later change is listed with its date, the reason, and whether confirmatory results had been seen at the time. No confirmatory hypothesis is added after the confirmatory networks are examined.

### 10.2 Freeze record

| Item | Status |
| --- | --- |
| Freeze date and commit | 21 November 2026, tag `sections67-protocol-v1.1` |
| External archive of the frozen version | Pending |
| Seeds and analysis samples per network | Recorded by the pipeline |
| Configurations and software environment | Recorded by the pipeline |
| Q/I_int and Sobol resolution parameters | Decomposition ±0.004 of Var(g); S1 and ST ±0.005 (§4.2, §6.2) |
| Platform tolerances | Exact reproduction expected; RelMSE differences < 1e-5 not investigated (§7.1) |
| Benchmark reference release | `v1.0-benchmark`, https://doi.org/10.5281/zenodo.24425847 |
| Confirmatory sample | Drawn with commit ca7fbdc before any simulation: `protocols/confirmatory_network_manifest.csv` (320 networks, 80 per set, no exclusions) and `protocols/confirmatory_candidate_permutation.csv` (full permutation with substitutes) |
| Start of the confirmatory analysis | Not started |

### 10.3 Deviations and incidents

**21 November 2026 — prior exposure of one confirmatory network.** During testing of the simulation pipeline, network j3042_8 (J30; first in the J30 permutation and part of the sample) was simulated, and the `structure` stage and the OLS, EN-36-main and EBM-24-noint fits were run. The following values were observed: C = 0, a single critical set, N_eff = 1 and OLS RelMSE ≈ 2.4e-28. No EN-36, EBM-24, fastai-41 or EN-36-centred fits, local effects or decisions were computed, so no input to H1–H4 was seen. The outputs were deleted. The network remains in the sample, because excluding or replacing it would depend on an observed result. Subsequent pipeline tests use networks outside the confirmatory sample. Because simulation is seeded (§9.2), the confirmatory run regenerates the same scenarios for this network.

**21 November 2026 — pre-freeze amendment after the technical pilot (version 1.1).** The pilot (27 exploratory networks; `protocols/pilot_report.md`) led to four technical decisions: (1) the platform tolerance of §7.1; (2) the Sobol settings of §6.2, kept as specified, with a declared resolution of ±0.005; (3) the resolution checks of §4.2 and §6.2, implemented as the `resolution` stage and run on five pilot networks, giving a declared resolution of ±0.004 of Var(g) for the decomposition; (4) the tolerance of the secondary violation measure of §6.1, changed from 1e-6 to 0.05, with the mean excess added, because at 1e-6 it counted numerical noise in 34–47% of perturbations. The pilot was then rerun in full with the amended code (commit 9ec86e3) and reproduced the first run exactly. The research questions, hypotheses H1–H4, their statistics, decision criteria, the Holm family and the confirmatory sample are unchanged. Pilot values of the hypothesis statistics were computed only on exploratory networks and did not motivate any of these changes.
