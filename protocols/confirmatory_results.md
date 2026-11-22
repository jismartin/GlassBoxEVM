# Confirmatory results for Sections 6–7

## 1. Provenance

| Item | Value |
| --- | --- |
| Protocol | `protocols/sections_6_7.md`, version 1.1 |
| Frozen version | Git tag `sections67-protocol-v1.1` (annotated tag object `c150c54`) on commit `a8db487`; tag created 2026-11-21 23:12 UTC (Git timestamp, set by the authors' machine) |
| Public availability of the frozen version | The cluster fetched the tag from the public GitHub repository at 2026-11-21 23:14:00 UTC (cluster file timestamp of the fetch) |
| External archive of the frozen version | None before the campaign; the frozen protocol is archived with the release of the Sections 6–7 work |
| Pipeline code | `scripts/sections67` at commit `a8db487` (unchanged since `9ec86e3`, which produced the final pilot); configuration hash `e5a034d7…` |
| Confirmatory sample | `protocols/confirmatory_network_manifest.csv` (320 networks, 80 per set), drawn with commit `ca7fbdc` and recorded in commit `601cf24` before any confirmatory network was simulated; regenerated on the cluster with the frozen code and verified identical |
| Benchmark reference | `v1.0-benchmark`, https://doi.org/10.5281/zenodo.24425847 |
| Execution | Altamira, one single-core SLURM task per network, submitted 2026-11-21 23:17:46 UTC (SLURM accounting); aggregation finished 2026-11-22 06:18 UTC |

The freeze is therefore an internal, version-controlled freeze whose public availability before the campaign is documented by the cluster's own records; it is not an external preregistration.

**Study history.** Before the freeze, one confirmatory network (j3042_8) was used during pipeline testing (protocol §10.3): its scenarios were simulated, the structure stage was run, and OLS, EN-36-main and EBM-24-noint were fitted. C, the number of critical sets, N_eff and the OLS RelMSE were inspected; C is an input to H1 and H2. The focal-model error differences, local-effect outcomes and decision outcomes used to test H1–H4 were not computed or inspected. The outputs were deleted and the network was kept in the sample. The protocol was amended once before the freeze, after the technical pilot (version 1.1, §10.3). **During the confirmatory campaign** there were no deviations: all 320 networks completed every stage, all 2,240 fits succeeded at the first attempt, reloaded models reproduced their predictions (maximum difference 2.8e-14), every network had at least three eligible activities, and no network or pair was lost from any contrast.

## 2. Primary hypothesis H1

| Slope of the paired difference on C (subset intercepts) | Estimate | 95% bootstrap interval |
| --- | --- | --- |
| RelMSE EN-36 − fastai-41 | 0.070 | [0.060, 0.080] |
| RelMSE EBM-24 − fastai-41 | 0.074 | [0.057, 0.093] |

Both lower bounds are greater than zero, so **H1 is confirmed**: the relative position of fastai-41 improves as critical-set competition C increases, against both interpretable models. The descriptive subset-specific slopes are all positive (EN−fastai: J30 0.089, J60 0.066, J90 0.059, J120 0.070; EBM−fastai: J30 0.079, J60 0.066, J90 0.061, J120 0.089). A positive slope does not mean that fastai-41 is more accurate in every network: EN-36 has a lower RelMSE than fastai-41 in 64.4% of networks (206/320) and EBM-24 in 68.8% (220/320), whereas fastai-41 has the lower mean RelMSE. The relation is an association across networks, not a demonstration that C causes the change.

## 3. Secondary hypotheses H2–H4 (Holm, family-wise level 0.05)

| Hypothesis | Statistic | Estimate | 95% interval | p | Holm p | Result |
| --- | --- | --- | --- | --- | --- | --- |
| H2 | β_J30 − β_J120 for Delta_EN−EBM on C | 0.030 | [−0.001, 0.062] | 0.060 | 0.060 | Not confirmed |
| H3 | Mean EN−EBM primary local-effect error (MAE_D) | −0.034 | [−0.039, −0.030] | 0.0001 | 0.0003 | Confirmed |
| H4 | Mean EN−EBM regret at k = 3 (duration units) | 0.0013 | [0.0005, 0.0021] | 0.0014 | 0.0028 | Confirmed |

The p-values are from the null-centred bootstrap with 10,000 replicates (minimum attainable value about 1e-4). The intervals are marginal, not Holm-adjusted.

### H2

**H2 is not confirmed.** The confirmatory least-squares contrast is 0.030 with interval [−0.001, 0.062]. The subset slopes of Delta_EN−EBM on C (J30 0.011, J60 0.000, J90 −0.002, J120 −0.019) have the direction seen in the exploratory networks (J30 0.028, J120 −0.033), and the contrast is about half the exploratory estimate (0.061).

The prespecified sensitivity analyses do not replace the confirmatory test. The Theil–Sen contrast, 0.034 [0.018, 0.051], has the same sign and more than half the magnitude, which meets the protocol's reading rule for a pattern that is robust to the estimator; this agreement is consistent with, but does not demonstrate, a pattern that is not driven by a few networks. Influence is appreciable: 22 networks have Cook's D > 4/n, and removing a single network changes the contrast by up to 0.008, about 26% of the estimate.

### H3

**H3 is confirmed.** EN-36 has a lower MAE_D than EBM-24 in all 320 networks. MAE_D measures the mean absolute discrepancy between surrogate and exact responses per unit of perturbation; the exact response lies between 0 and 1. The difference of −0.034 is in those units; it is not a 3.4% reduction in error. The advantage of EN-36 is not confined to nearly linear networks: in the networks with C > 0, the mean MAE_D is 0.046 for EN-36 against 0.086 for EBM-24.

| MAE_D | EN-36 mean / median | EBM-24 mean / median | EN−EBM mean / median |
| --- | --- | --- | --- |
| All networks (320) | 0.031 / 0.007 | 0.065 / 0.032 | −0.034 / −0.024 |
| C = 0 (107) | 0.0003 / 0.0003 | 0.023 / 0.014 | −0.023 / −0.014 |
| C > 0 (213) | 0.046 / 0.042 | 0.086 / 0.083 | −0.040 / −0.037 |

The median ratio of EN-36 to EBM-24 MAE_D is 0.20. This result establishes greater fidelity of EN-36 to the exact local effects under the perturbations studied (±0.5 and ±1 training standard deviations). By itself it does not establish better human understanding of the explanations or better decisions.

### H4

**H4 is confirmed, with a difference favourable to EBM-24.** Under the median-cap intervention, EBM-24 achieved a statistically detectable but very small average advantage over EN-36: the greedy policy guided by EBM-24 achieves a reduction in mean duration 0.0013 units larger than the policy guided by EN-36. In context:

- relative to the benefit of the intervention (exact-greedy reduction at k = 3: mean 0.673, median 0.687), the difference is 0.19% of the mean benefit;
- relative to project duration (baseline mean duration: mean 83.3, median 85.3), it is 0.0015% (ratio of means) or 0.0019% (mean of per-network ratios).

The regret difference is zero in 253 networks (79%), EN-36 has the larger regret in 42 and EBM-24 in 25. In all 253 ties, EN-36 and EBM-24 select the same set of activities. As prespecified, the results are also reported by competition:

| Networks | n | Mean EN−EBM regret | EN larger / EBM larger / equal | Mean regret EN / EBM / fastai |
| --- | --- | --- | --- | --- |
| C = 0 | 107 | −0.0001 | 5 / 3 / 99 | 0.0000 / 0.0001 / −0.0002 |
| C > 0 | 213 | 0.0020 | 37 / 22 / 154 | 0.0025 / 0.0005 / 0.0004 |

**Selected activities at k = 3** (number of networks with identical sets):

| Networks | n | EN = EBM | EN = exact | EBM = exact | fastai = exact | SSI = exact | CI = exact |
| --- | --- | --- | --- | --- | --- | --- | --- |
| All | 320 | 253 | 275 | 284 | 265 | 246 | 3 |
| C = 0 | 107 | 99 | 107 | 99 | 90 | 91 | 0 |
| C > 0 | 213 | 154 | 168 | 185 | 175 | 155 | 3 |

In the C = 0 networks, EN-36 selects the exact-greedy set in all 107 networks, EBM-24 differs in 8 and fastai-41 in 17. A negative regret means that a surrogate's set performed better than the exact-greedy set on the evaluation sample, which the protocol allows. The H4 difference arises in the networks with critical-set competition. Over all networks, EN-36 selects the exact-greedy set in 85.9% of networks (275/320) and EBM-24 in 88.8% (284/320).

**Comparison with the static SSI ranking (descriptive).** As ratios of mean benefits at k = 3, relative to exact-greedy, the policies obtain: EN-36 99.75%, EBM-24 99.94%, fastai-41 99.97% and the static SSI ranking 99.75% (P90: 99.64%, 99.95%, 99.90% and 99.64%). The surrogate-guided policies therefore work well, but a simple ranking by the schedule sensitivity index obtains a very similar benefit. In this intervention, the test provides limited evidence of decision value of the surrogates beyond SSI; it does not establish statistical equivalence between policies. The static CI ranking obtains 66.1% of the exact-greedy benefit. This should not be read as a general failure of the criticality index: CI measures how often an activity is critical, not how much its duration can be reduced by the median cap, and when several activities have CI = 1 the rule breaks ties by the smallest activity identifier, which can select activities with very different benefits.

## 4. Additional sensitivity analysis (not prespecified)

The protocol keeps j3042_8 in the sample and did not prespecify an analysis without it. As an additional check, the inference was repeated without j3042_8 (319 networks): H1 slopes 0.070 [0.060, 0.081] and 0.074 [0.057, 0.093]; H2 0.030 [−0.001, 0.063], Holm p 0.062; H3 −0.034 [−0.039, −0.031], Holm p 0.0003; H4 0.0013 [0.0005, 0.0021], Holm p 0.003. All conclusions are unchanged.

## 5. Descriptive comparisons (not confirmatory)

| Paired difference in RelMSE | Mean | Median of paired differences | First model has lower RelMSE |
| --- | --- | --- | --- |
| EN-36 − EBM-24 | −0.0006 | −0.00007 | 71% |
| EN-36 − fastai-41 | 0.0030 | −0.0028 | 64% |
| EBM-24 − fastai-41 | 0.0036 | −0.0031 | 69% |
| OLS − EN-36 | 0.0248 | 0.00006 | 43% |
| EBM-24 − EBM-24-noint | −0.0221 | −0.00009 | 94% |
| EN-36 − EN-36-main | −0.0248 | −0.00006 | 59% |
| EN-36 − EN-36-centred | −0.0075 | −0.00003 | 59% |

EN-36 and EBM-24 have a lower RelMSE than fastai-41 in 64% and 69% of networks, whereas fastai-41 has the lower mean RelMSE. The ablations describe the retained configurations with one component removed or changed, without retuning: removing the pairwise terms of EBM-24 or the product terms of EN-36 increases the mean RelMSE by about 0.022–0.025. For EN-36 this gain may combine non-linear representation with a different penalisation of nearly linear effects. The comparison with EN-36-centred (mean RelMSE 0.0075 higher, 41% of networks better) does not show that centring is harmful in itself, because its penalty was not retuned.

## 6. Summary

- **H1 confirmed.** The relative position of the flexible neural network improves with critical-set competition against both interpretable models, while the interpretable models still have lower error in most individual networks.
- **H2 not confirmed.** The J30–J120 difference in how EN−EBM depends on C has the exploratory direction and about half its size; the confirmatory test is not significant, and single networks have appreciable influence.
- **H3 confirmed.** Elastic Net reproduces the local effects of the exact function more faithfully than EBM in every network.
- **H4 confirmed, with a difference favourable to EBM.** Under the median-cap intervention, EBM-24 achieved a statistically detectable but very small average advantage over EN-36, amounting to 0.19% of the exact-greedy mean benefit. Both models usually select the same activities as exact-greedy, and the static SSI ranking obtains a very similar benefit.

## Appendix A. Intervention benefits by policy

Mean benefit in mean duration (B_mean) and in the 90th percentile (B_p90), and regret relative to exact-greedy, over 320 networks (exhaustive search: J30 and J60 only, 160 networks). Units are duration units.

| Policy | k | B_mean mean | B_mean median | B_p90 mean | Regret mean | Regret median |
| --- | --- | --- | --- | --- | --- | --- |
| Exact-greedy | 1 | 0.2429 | 0.2502 | 0.3641 | 0 | 0 |
| | 2 | 0.4679 | 0.4771 | 0.6989 | 0 | 0 |
| | 3 | 0.6731 | 0.6871 | 1.0052 | 0 | 0 |
| EN-36 greedy | 1 | 0.2426 | 0.2503 | 0.3633 | 0.0003 | 0 |
| | 2 | 0.4670 | 0.4768 | 0.6974 | 0.0009 | 0 |
| | 3 | 0.6714 | 0.6853 | 1.0016 | 0.0017 | 0 |
| EBM-24 greedy | 1 | 0.2430 | 0.2504 | 0.3639 | −0.0001 | 0 |
| | 2 | 0.4677 | 0.4771 | 0.6991 | 0.0003 | 0 |
| | 3 | 0.6727 | 0.6871 | 1.0047 | 0.0004 | 0 |
| fastai-41 greedy | 1 | 0.2421 | 0.2491 | 0.3624 | 0.0009 | 0 |
| | 2 | 0.4671 | 0.4767 | 0.6975 | 0.0008 | 0 |
| | 3 | 0.6729 | 0.6868 | 1.0042 | 0.0002 | 0 |
| SSI static | 1 | 0.2413 | 0.2477 | 0.3607 | 0.0016 | 0 |
| | 2 | 0.4660 | 0.4760 | 0.6958 | 0.0019 | 0 |
| | 3 | 0.6714 | 0.6854 | 1.0015 | 0.0017 | 0 |
| CI static | 1 | 0.1187 | 0.1081 | 0.1596 | 0.1242 | 0.1336 |
| | 2 | 0.2865 | 0.2925 | 0.3950 | 0.1814 | 0.1795 |
| | 3 | 0.4452 | 0.4586 | 0.6156 | 0.2279 | 0.2163 |
| Exhaustive (sample optimum) | 1 | 0.2425 | 0.2495 | 0.3725 | 0.0000 | 0 |
| | 2 | 0.4651 | 0.4745 | 0.7129 | 0.0000 | 0 |
| | 3 | 0.6650 | 0.6825 | 1.0199 | −0.0001 | 0 |

The three surrogate-guided policies and the static SSI ranking all obtain almost the full exact-greedy benefit, whereas the static CI ranking loses about a third of it at k = 3. P90 benefits show a broadly similar pattern, although the ordering of policies is not identical (at k = 3, fastai-41 has the larger mean benefit and EBM-24 the larger P90 benefit). Selection optimised mean duration, not P90. The exhaustive rows cover only the 160 J30 and J60 networks, so their benefit means are not comparable with the other rows; their regrets are computed against exact-greedy on the same networks.
