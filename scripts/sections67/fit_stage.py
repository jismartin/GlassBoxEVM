"""Stage 'fit': one task per network and model; persists pipeline, metadata and predictions."""
import os
import time
import traceback

import numpy as np
import pandas as pd

from . import common
from . import config as cfg
from . import data
from . import models


def stored_benchmark_relmse(ctx, network, model_id):
    """RelMSE of the historical benchmark result file, if it exists (EN-36, EBM-24, fastai-41)."""
    spec = cfg.MODELS[model_id]
    if model_id not in ("EN-36", "EBM-24", "fastai-41"):
        return None
    subset = ctx.subset_of(network)
    path = os.path.join(ctx.repo, "model_evaluation", "results",
                        "model_evaluation_%s_lhs_%s_subset_%s_project_%s.csv"
                        % (spec["family"], spec["lhs"], subset, network))
    if not os.path.exists(path):
        return None
    return float(pd.read_csv(path, float_precision="round_trip")["RelMSE"].iloc[0])


def run(ctx, network, model_id):
    out = ctx.stage_dir(network, "fit", model_id)
    inputs = {"scenarios_sha256": data.scenario_input_hash(ctx, network),
              "test_runs_sha256": common.sha256_file(os.path.join(ctx.repo, cfg.SIMULATOR["test_runs_file"])),
              "model_id": model_id}
    st = common.Stamp(out, inputs)
    if not st.begin(ctx.force):
        return None
    df, dcols, _ = data.load_scenarios(ctx, network)
    df_train, df_test = data.split(df, ctx.test_runs)
    const = data.constant_mask(df_train, dcols)
    attempts = 0
    errors = []
    pipe = info = None
    t0 = time.time()
    while attempts < 2 and pipe is None:            # one retry with the same design (§7)
        attempts += 1
        try:
            pipe, info = models.fit(ctx.repo, model_id, df_train, df_test, dcols, const)
        except Exception:  # noqa: BLE001
            errors.append(traceback.format_exc())
    runtime = time.time() - t0
    row = {"cohort": ctx.cohort, "network_id": network, "subset": ctx.subset_of(network),
           "model_id": model_id, "config_hash": cfg.config_hash(), "attempts": attempts,
           "runtime": runtime, "memory": common.peak_memory_mb()}
    if pipe is None:
        common.atomic_write_text(os.path.join(out, "fit_errors.txt"), "\n\n".join(errors))
        row.update(fit_status="failed", model_path="", RelMSE=np.nan)
        common.atomic_write_csv(pd.DataFrame([row]), os.path.join(out, "fit_manifest.csv"))
        return st.finish(status="failed", info={"errors": len(errors)})

    model_path = os.path.join(out, models.model_filename(model_id))
    models.save(pipe, model_id, model_path)
    X_all = df[dcols].to_numpy(float)
    pred_all = pipe.predict(X_all)
    test_pos = df.index.get_indexer(ctx.test_runs)
    is_test = np.zeros(len(df), bool)
    is_test[test_pos] = True
    y = df["actual_duration"].to_numpy(float)
    ytr = df_train["actual_duration"].to_numpy(float)
    yte = y[test_pos]
    # protocol formula (float64 SSE ratio) on the persisted-pipeline predictions
    rel_sse = float(np.sum((yte - pred_all[test_pos]) ** 2) / np.sum((yte - ytr.mean()) ** 2))
    fit_path_diff = float(np.max(np.abs(info["test_pred_fit_path"] - pred_all[test_pos])))
    # reload check (§3.3, §7.2)
    reload_pipe, _ = None, None
    meta = {"network_id": network, "model_id": model_id, "column_order": dcols,
            "constant_mask": const.tolist(),
            "input_columns": getattr(pipe, "keep_cols", [c for c, k in zip(dcols, ~const) if k] if
                                     model_id == "OLS" else dcols),
            "test_indices": [int(i) for i in ctx.test_runs],
            "train_indices_rule": "df.index.difference(test_runs)",
            "seed": cfg.SEEDS["training_seed"], "config_hash": cfg.config_hash(),
            "code_hash": common.code_hash(), "environment": common.environment_info(),
            "hyperparameters_used": info["hyperparameters_used"],
            "RelMSE_benchmark_arithmetic": info["RelMSE"], "RelMSE_sse_float64": rel_sse,
            "fit_time_s": info["fit_time_s"],
            "n_terms": info.get("n_terms"), "n_pair_terms": info.get("n_pair_terms"),
            "epochs_run": info.get("epochs_run"), "n_constant_dropped": info.get("n_constant_dropped")}
    common.atomic_write_json(os.path.join(out, "model_meta.json"), meta)
    reload_pipe, _ = models.load(out, model_id)
    pred_reload = reload_pipe.predict(X_all[test_pos])
    reload_diff = float(np.max(np.abs(pred_reload - pred_all[test_pos])))
    stored = stored_benchmark_relmse(ctx, network, model_id)
    pred = pd.DataFrame({"cohort": ctx.cohort, "network_id": network, "model_id": model_id,
                         "scenario_id": df.index.to_numpy(), "split": np.where(is_test, "test", "train"),
                         "y_true": y, "y_pred": pred_all})
    pred_path = common.write_table(pred, os.path.join(out, "predictions"))
    status = "ok" if reload_diff <= 1e-9 * max(1.0, float(np.abs(pred_all).max())) else "reload_mismatch"
    row.update(fit_status=status, model_path=model_path, RelMSE=info["RelMSE"], RelMSE_sse=rel_sse,
               RelMSE_stored=stored if stored is not None else np.nan,
               RelMSE_rel_diff_stored=(info["RelMSE"] / stored - 1) if stored else np.nan,
               reload_max_abs_diff=reload_diff, fitpath_vs_pipeline_max_abs_diff=fit_path_diff,
               fit_time_s=info["fit_time_s"], n_terms=info.get("n_terms"),
               n_pair_terms=info.get("n_pair_terms"), predictions_path=pred_path,
               code_hash=common.code_hash())
    common.atomic_write_csv(pd.DataFrame([row]), os.path.join(out, "fit_manifest.csv"))
    return st.finish(status=status, info={"RelMSE": info["RelMSE"], "reload_max_abs_diff": reload_diff})
