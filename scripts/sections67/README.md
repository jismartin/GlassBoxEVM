# scripts/sections67 — Sections 6–7 pipeline

Implements protocol `protocols/sections_6_7.md` (v1.0), §§2–9. The simulator
(`scripts/tools.py`), `model_evaluation/model_evaluation.py`, `model_selection/`, the
historical results and the retained configurations (EN LHS 36, EBM LHS 24, fastai LHS 41 with
the constant-column drop "fix A") are used read-only and never modified.

Requires Python ≥ 3.9 with the `mlenv` versions (numpy 2.0.2, pandas 2.3.2, scipy 1.13.1,
scikit-learn 1.6.1, interpret 0.7.2, torch 2.8.0 CPU, fastai 2.7.19, networkx 3.2.1, joblib).
`pyarrow` is optional: without it the detail tables are written as `.csv.gz` instead of Parquet.

## Command line

Run from the repository root:

```
python -m scripts.sections67 <stage> --cohort <c> [--network <id>|all|id1,id2 | --network-index <i>]
                             [--model <m>[,<m2>...]] [--results DIR] [--threads N] [--force]
```

- `--results` (default `$SECTIONS67_RESULTS`, else `<repo>/../GlassBoxEVM_sections67_results`)
  must lie outside `scripts/`. All outputs go there; nothing is written into the repository.
- `--network-index i` picks the i-th *included* network of `<results>/<cohort>/network_manifest.csv`
  (0-based), for SLURM array tasks.
- `--threads` (default `$SLURM_CPUS_PER_TASK` or 1) limits OpenMP/BLAS before numpy is imported;
  EBM `n_jobs` and fastai torch threads follow `SLURM_CPUS_PER_TASK`, as in model_evaluation.py.
- `python -m scripts.sections67 config` prints the configuration and its SHA-256 hash.

| stage | unit of work | what it does |
| --- | --- | --- |
| `prepare` | cohort | `network_manifest.csv` (sha256 of the `.sm` file, seeds, status, exclusion reason); for `--cohort confirmatory` also `candidate_permutation.csv` (one `default_rng([20261119, k])` per subset, full permutation, first 80 valid selected, rest = ordered substitutes). `exploratory` = the 360 evaluation networks; any other cohort needs `--network-list nets.csv` (column `network_id`), e.g. the 24-network pilot. |
| `simulate` | network | control point via `tools.add_risk_parameters` + `tools.evaluate_at_EV`; seeded beta-PERT scenarios (seed = SHA-256 rule of §9.2). New networks: 10,000 train/test scenarios (stage `simulate`). All networks: `decision_select` (5,000), `decision_eval` (5,000), `sobol_sample` (10,000); `support.csv` with the theoretical support of each column (in-progress activities: `elapsed + (1-done)·[O,P]`). The 400 original networks keep `montecarlo/<id>_simulation_01.csv`. |
| `fit` | network × model | EN-36, EBM-24, fastai-41 (code copied from model_evaluation.py), OLS, EBM-24-noint, EN-36-main, EN-36-centred. Saves the full pipeline (`model.joblib`, or fastai `model.pkl` via `Learner.export`), `model_meta.json` (column order, constant mask, test indices, hyperparameters, environment), `predictions.*`, `fit_manifest.csv` with RelMSE (benchmark arithmetic), RelMSE as an SSE ratio, the stored benchmark value and the reload check. One retry after a technical failure. |
| `structure` | network | exact g check against `actual_duration`, C (tolerance 1e-9·T; sensitivity at 1e-7 and 1e-5), number of critical sets, N_eff, CI per activity, share of variable activities with 0.05<CI<0.95, SP/TF and the other indicators of `tools.compute_topological_indicators`, out-of-sample OLS RelMSE, and the empirical L_emp/I_int_emp/Q_emp (`--no-decomposition` to skip). |
| `explain` | network × model | `--model exact` first (cached reference for g), then each surrogate: local finite differences (500 test scenarios, ±0.5/±1 training SD, clipped to the theoretical support, exclusion if \|δ_ef\| < 10 % of the step), PD error on 21 quantiles with a 500-test background, S1 (PD variance) and ST (Jansen, 3 permutation seeds) on `sobol_sample`. |
| `decide` | network × model | `--model exact`: exact-greedy, CI and SSI static rankings, exhaustive search for J30/J60 (k = 1, 2, 3); `--model <m>`: surrogate-greedy. Median cap on the selection sample, evaluation with g on `decision_eval` (B_mean, B_p90, expected duration reduction, cap check). |
| `aggregate` | cohort | collects per-network files, builds `network_metrics.csv`, and writes `inference.csv` (H1 IUT criterion, H2 with Theil–Sen and influence diagnostics in `h2_influence.csv`, H3, H4, Holm over H2–H4, descriptive pairs and ablations) and `run_manifest.json`/`run_manifest.md`. |

Default models: `fit` all seven; `explain` exact + all seven; `decide` exact + EN-36, EBM-24, fastai-41.

## Outputs (file contract of §9.6)

```
<results>/<cohort>/network_manifest.csv, candidate_permutation.csv, config_snapshot.json
<results>/<cohort>/networks/<id>/simulate/{train_test,decision_select,decision_eval,sobol_sample}.parquet|csv.gz, support.csv
<results>/<cohort>/networks/<id>/fit/<model>/{model.joblib|model.pkl, model_meta.json, predictions.*, fit_manifest.csv}
<results>/<cohort>/networks/<id>/structure/{structure_metrics,decomposition,activity_structure,critical_sets}.csv
<results>/<cohort>/networks/<id>/explain/<model>/{local_metrics.csv, local_detail.*, sensitivity.*, pd_curves.*}
<results>/<cohort>/networks/<id>/explain/exact/{exact_reference.npz, indices_*.csv}
<results>/<cohort>/networks/<id>/decide/<model>/{decisions.csv, medians.csv}
<results>/<cohort>/aggregate/{fit_manifest,structure_metrics,decomposition,local_metrics,decisions,network_metrics,inference,h2_influence}.csv, sensitivity.*
<results>/<cohort>/run_manifest.json, run_manifest.md
```

Every stage directory carries `_stamp.json` (config hash, code hash, input hashes, runtime, peak
memory, status). Outputs are written to a temporary file and renamed; the stamp is written last.
A stage is skipped when its stamp matches; if inputs, configuration or code changed, the old
directory is moved to `<dir>.superseded-<UTC>` and the stage is recomputed (never overwritten
silently). Exploratory, pilot and confirmatory cohorts live in separate folders.

## Running on Altamira as array tasks (no shell scripts in the repository)

Use one SLURM array per stage (and per model for `fit`, `explain`, `decide`), with
`--cpus-per-task=1` for fits (benchmark timing budget) and the array range `0-(N-1)` where N is
the number of included networks. The partition and account must be taken from the cluster
configuration; they are not set here. Each array task runs one command, for example:

```
python -m scripts.sections67 prepare   --cohort pilot --network-list pilot_networks.csv --results $RES   # once
python -m scripts.sections67 simulate  --cohort pilot --network-index $SLURM_ARRAY_TASK_ID --results $RES
python -m scripts.sections67 structure --cohort pilot --network-index $SLURM_ARRAY_TASK_ID --results $RES
python -m scripts.sections67 fit       --cohort pilot --network-index $SLURM_ARRAY_TASK_ID --model EN-36 --results $RES
python -m scripts.sections67 explain   --cohort pilot --network-index $SLURM_ARRAY_TASK_ID --model exact --results $RES
python -m scripts.sections67 explain   --cohort pilot --network-index $SLURM_ARRAY_TASK_ID --model EN-36 --results $RES
python -m scripts.sections67 decide    --cohort pilot --network-index $SLURM_ARRAY_TASK_ID --model exact --results $RES
python -m scripts.sections67 decide    --cohort pilot --network-index $SLURM_ARRAY_TASK_ID --model EN-36 --results $RES
python -m scripts.sections67 aggregate --cohort pilot --results $RES                                    # once
```

Dependencies: `simulate` → (`structure`, `fit`) → `explain --model exact` → `explain --model <m>`;
`simulate` + `fit` → `decide --model exact` → `decide --model <m>` (the model task needs only
its own fit, but run `exact` first so that two tasks never build the same cache). Use
`--dependency=aftercorr:<jobid>` between arrays. A failed task returns exit code 1 and can be
resubmitted; finished tasks are skipped through their stamps.

## Tests

```
SECTIONS67_TEST_RESULTS=<results> SECTIONS67_TEST_COHORT=localtest \
SECTIONS67_C_REFERENCE=<csv with project,C> python -m pytest scripts/sections67/tests -v
```

Unit checks (seed rule, Holm example, null-centred p, analytic S1 = ST on a linear function,
row-order invariance, cap property) need no data; data checks (partitions, reload, EN benchmark
reproduction, exact D in [0, 1], monotonicity of g, cap never increases g, exhaustive ≤ greedy,
C reference, simulator distributional equivalence) read a local run.

## Declared deviations from the protocol text

- `configs/sections_6_7.yaml` is replaced by `config.py` (YAML/JSON cannot be stored in the
  repository with the available tooling); `config_hash()` is the SHA-256 of its canonical JSON.
- `run_manifest` is written both as JSON (results directory) and as Markdown.
- Detail tables are Parquet only when pyarrow exists; otherwise gzip CSV.
- S1 uses a 500-row background drawn from `sobol_sample` (to be fixed in the pilot).
- EN-36 RelMSE reproduces the stored Altamira values to ≤ 1e-12 relative, not always bit for
  bit on another platform; fastai-41 is not reproducible across machines (known, documented in
  the benchmark audit); the platform tolerance is fixed in the pilot.
