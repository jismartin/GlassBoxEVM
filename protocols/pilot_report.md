# Technical pilot of the Sections 6–7 pipeline

**Protocol:** `protocols/sections_6_7.md`, version 1.0 (commit `ca7fbdc`).
**Code:** `scripts/sections67`, first run at commit `ca7fbdc` (configuration hash `f9f4a383…`); final run at commit `9ec86e3` (configuration hash `e5a034d7…`), after the amendments listed in §8.
**Platform:** Altamira (IFCA), single-core SLURM array tasks, Python 3.9.23 with the benchmark environment (`mlenv`).
**Date:** 21 November 2026.

The final run at commit `9ec86e3` reproduced the first run exactly: all 189 fit RelMSE values, all 216 primary local errors and all 525 decision benefits are identical. The timings in §2 are from the first run; all other values are from the final run, and coincide with the first run wherever both exist.

## 1. Networks

The pilot uses 27 exploratory networks; no confirmatory network was simulated or fitted. Selection used only C and the OLS deviation, never model performance:

- **24 pilot networks.** In each PSPLIB set: the C = 0 network with the median OLS deviation, and the five C > 0 networks closest to the 10, 30, 50, 70 and 90% quantiles of C (ties broken by identifier).
- **3 illustrative networks (protocol §8).** j9040_3 (C = 0, median OLS deviation), j902_5 (two critical sets, C closest to 0.5) and j3025_2 (maximum C).

All refits use the stored benchmark scenarios (`montecarlo/`) and the original partition (`montecarlo/test_runs.csv`).

## 2. Execution

Both runs completed every task (568 in the first run; 573 in the final run, which adds the five `resolution` tasks): 27 networks × (simulation, structure, 7 fits, 8 explanation runs, 4 decision runs) plus aggregation. There were no failed tasks and no retries.

| Stage | CPU-hours (27 networks) | Longest task (s) |
| --- | --- | --- |
| Fit EBM-24 | 6.31 | 2,503 |
| Fit EN-36 | 4.75 | 3,206 |
| Fit EBM-24-noint | 1.57 | 927 |
| Explain EN-36 / EN-36-centred | 1.55 / 1.55 | 551 / 527 |
| Decide exact (incl. exhaustive J30/J60) | 1.19 | 651 |
| Fit fastai-41 | 0.53 | 85 |
| Structure | 0.52 | 149 |
| All other stages | 2.13 | ≤ 154 |
| **Total** | **20.1** | |

The peak memory of a fit was 2.3 GB (EN-36 and EN-36-centred, J120); other models stayed below 0.5 GB. Median fit time on J120 was 1,534 s for EN-36 and 1,056 s for EBM-24. EN-36-centred converged much faster than EN-36 (median 12 s on J120).

**Sizing of the confirmatory run.** At 0.745 CPU-hours per network, the 320 confirmatory networks need about 240 CPU-hours, plus the simulation of their training and test scenarios. The pilot over-represents no set strongly (J30 7, J60 6, J90 8, J120 6), so this estimate is adequate for planning. The time limits used in the pilot (4 h for EN-36 and EN-36-centred, 3 h for EBM-24) leave a margin of at least 4× over the longest task.

## 3. Reproduction of the benchmark

| Model | Networks | RelMSE identical to the stored value | Maximum relative difference |
| --- | --- | --- | --- |
| EBM-24 | 27 | 27 | 0 |
| fastai-41 | 27 | 27 | 0 |
| EN-36 | 27 | 26 | 0.139 (j6017_10) |

EBM-24 and fastai-41 reproduce the benchmark bit for bit on Altamira. For EN-36, one network (j6017_10, C = 0.0007) gives RelMSE 2.707e-5 against a stored 2.376e-5, an absolute difference of 3.3e-6. Two further refits of this network on other compute nodes returned exactly 2.707e-5, so the refit is deterministic and the difference lies in the stored benchmark value; its origin could not be traced. The benchmark tables keep the stored values. The Sections 6–7 analyses use the refitted models.

**Platform tolerance.** On Altamira, refits are expected to reproduce the benchmark exactly. Any difference is reported per network, and differences in RelMSE below 1e-5 in absolute value are not investigated further.

## 4. Acceptance checks (protocol §7.2)

| Check | Result |
| --- | --- |
| Identifiers and partitions match | ✓ |
| Reloaded models reproduce predictions | ✓ (maximum difference 2.8e-14) |
| g reproduces the simulated makespan | ✓ (maximum difference 7.1e-14) |
| C equals the exploratory reference | ✓ (difference 0; tolerance 1e-7 identical; 1e-5 changes C by at most 0.0004) |
| Exact local effects D in [0, 1] | ✓ (range 0 to 1 + 3e-12) |
| Every activity has valid perturbations | ✓ |
| Capping never increases g | ✓ (maximum increase 0) |
| Exhaustive search ≤ greedy on the selection sample | ✓ (39 comparisons) |
| Empirical decomposition reproduces the exploratory values | ✓ (L_emp, I_int_emp, Q_emp within 4e-16) |
| Holm and null-centred bootstrap p-values on known examples | ✓ (local acceptance tests) |

## 5. Numerical resolution

The `resolution` stage was run on five networks chosen by rule (in each set, the pilot network with C > 0 nearest the 50% quantile of C: j3028_4, j601_4, j9023_2, j12049_8; plus the maximum-C network j3025_2).

**Sobol indices of g.** Across the three permutation seeds, the standard deviation of ST has median 0, 95th percentile 1.1e-3 and maximum 6.3e-3. In the five C = 0 networks, where S1 = ST exactly, |S1 − ST| has 95th percentile 6e-4 and maximum 2.8e-3, and ΣS1 lies between 0.984 and 1.015. A 41-point grid changes individual S1 values by at most 0.0006, and a 2,000-row background instead of 500 rows by at most 0.0049 (95th percentile 0.0007). The declared resolution of S1 and ST is ±0.005.

**Decomposition.** A 41-point grid changes Q_emp by at most 1.3e-5 and I_int_emp by at most 2e-5. The summed single-activity curvature agrees with Q_emp within 2.2e-4. Evaluated on an independent sample of 10,000 scenarios, L_emp, I_int_emp and Q_emp change by at most 1.8e-3, 1.8e-3 and 3.6e-3 of Var(g) (Q_emp of j3025_2: 0.030 in-sample, 0.034 independent). The declared resolution of the decomposition is ±0.004 of Var(g); it is limited by sampling, not by the grid.

| Network | C | L_emp | Q_emp (21) | Q_emp (41) | Curvature sum | Q (independent) |
| --- | --- | --- | --- | --- | --- | --- |
| j3028_4 | 0.052 | 0.0032 | 0.00075 | 0.00076 | 0.00074 | 0.00067 |
| j601_4 | 0.053 | 0.0155 | 0.00035 | 0.00035 | 0.00058 | 0.00063 |
| j12049_8 | 0.082 | 0.0153 | 0.00086 | 0.00085 | 0.00081 | 0.00080 |
| j9023_2 | 0.138 | 0.0131 | 0.00254 | 0.00255 | 0.00258 | 0.00262 |
| j3025_2 | 0.681 | 0.1819 | 0.03034 | 0.03034 | 0.03043 | 0.03398 |

**Partial-dependence error.** Relative to the 21-point grid, a 41-point grid changes the PD error by 6–15% for EBM-24 and by less than 1% for EN-36 and fastai-41. Adding the support limits to the grid raises the PD error of EBM-24 by 13–214% and of EN-36 by 2–14%, and leaves fastai-41 practically unchanged. The ordering of the three models is the same under the three settings in all five networks. The 21-point grid with linear extrapolation is kept; PD errors of EBM depend on the treatment of the support extremes, which is stated when they are reported.

## 6. Local-effect violation measure

At the original tolerance of 1e-6, the share of perturbations with D outside [0, 1] was 34–47% for EN-36, EBM-24 and fastai-41, because almost any finite-difference noise counted. With the amended tolerance of 0.05, the mean rates are OLS 0.014, EN-36-main 0.014, EN-36 0.036, EBM-24-noint 0.092, EBM-24 0.096, EN-36-centred 0.116 and fastai-41 0.187, and the mean excess of D outside [0, 1] ranges from 0.004 (OLS) to 0.049 (fastai-41). The primary error MAE_D is unaffected by the tolerance.

## 7. Indicative values for H3 and H4

These values come from 27 networks chosen to span C. They describe variability for planning only; they are not estimates of the confirmatory effects.

- **H3 (local error, EN−EBM).** Mean −0.036, median −0.024, SD 0.034; EN has the lower error in 27 of 27 networks. Median MAE_D: OLS 0.020, EN-36 0.025, EBM-24 0.057, fastai-41 0.101.
- **H4 (regret at k = 3, EN−EBM).** Mean −0.0017, median 0, SD 0.0043; the difference is exactly 0 in 23 of 27 networks. Regret is 0 for EN and EBM in four of the five C = 0 networks; in the fifth, EBM selects a different set (regret 0.009). EN-greedy sometimes outperforms exact-greedy on the evaluation sample (minimum regret −0.011).

## 8. Decisions taken before the freeze

1. **Platform tolerance.** Refits on Altamira are expected to reproduce the benchmark exactly; any difference is reported per network, and RelMSE differences below 1e-5 are not investigated further.
2. **Sobol settings.** The 10,000-scenario Sobol sample and the 500-row PD background for S1 are kept, with a declared resolution of ±0.005.
3. **Resolution checks.** Implemented as the `resolution` stage and run on five networks (§5); the decomposition has a declared resolution of ±0.004 of Var(g).
4. **Violation measure.** Tolerance changed from 1e-6 to 0.05, and the mean excess of D outside [0, 1] added (§6).
5. **Confirmatory run.** Time limits as in the pilot (4 h for EN-36 and EN-36-centred, 3 h for EBM-24, 1 h or less for the other models) and 8 GB per task (16 GB for EN-36 and EN-36-centred).

These decisions are technical. The hypotheses, statistics, decision criteria and confirmatory sample of the protocol were not changed, and the pilot values of the hypothesis statistics (§7), computed on exploratory networks only, did not motivate any of them. They are recorded in protocol version 1.1 (§10.3).
