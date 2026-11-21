"""Argument parsing and stage dispatch."""
import argparse
import os
import sys
import time
import traceback

from . import config as cfg

STAGES = ["prepare", "simulate", "fit", "structure", "explain", "decide", "resolution", "aggregate", "config"]
DEFAULT_MODELS = {"fit": cfg.ALL_MODELS, "explain": ["exact"] + cfg.ALL_MODELS,
                  "decide": ["exact"] + cfg.FOCAL_MODELS}


def build_parser():
    from . import data
    p = argparse.ArgumentParser(prog="python -m scripts.sections67",
                                description="GlassBoxEVM Sections 6-7 pipeline (protocol v%s)" % cfg.PROTOCOL_VERSION)
    p.add_argument("stage", choices=STAGES)
    p.add_argument("--cohort", default="exploratory",
                   help="output cohort label: exploratory, confirmatory, pilot, ... (separate folders)")
    p.add_argument("--network", default=None,
                   help="network id, comma-separated ids, or 'all' (= included networks of the manifest)")
    p.add_argument("--network-index", type=int, default=None,
                   help="0-based position in the included networks of the manifest (SLURM array task id)")
    p.add_argument("--network-list", default=None, help="CSV with network_id column (prepare, custom cohorts)")
    p.add_argument("--model", default=None,
                   help="model id or comma list (%s; 'exact' for explain/decide)" % ", ".join(cfg.ALL_MODELS))
    p.add_argument("--repo", default=data.default_repo(), help="repository root (inputs, read-only)")
    p.add_argument("--results", default=None,
                   help="results directory (default: $SECTIONS67_RESULTS or <repo>/../GlassBoxEVM_sections67_results)")
    p.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    p.add_argument("--force", action="store_true", help="recompute even if a matching stamp exists")
    p.add_argument("--no-decomposition", action="store_true", help="structure: skip Q/I_int")
    p.add_argument("--n-boot", type=int, default=None, help="aggregate: bootstrap replicates (default config)")
    return p


def _networks(ctx, args):
    from . import prepare
    if args.network_index is not None:
        return [prepare.included_networks(ctx)[args.network_index]]
    if args.network in (None, "all"):
        return prepare.included_networks(ctx)
    return [s.strip() for s in args.network.split(",") if s.strip()]


def _models(args):
    if args.model:
        ms = [s.strip() for s in args.model.split(",")]
    else:
        ms = DEFAULT_MODELS[args.stage]
    for m in ms:
        if m != "exact" and m not in cfg.MODELS:
            raise SystemExit("unknown model %s" % m)
    return ms


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.stage == "config":
        print(cfg.config_json())
        print("config_hash", cfg.config_hash())
        return 0
    from . import data
    results = args.results or data.default_results(args.repo)
    ctx = data.Ctx(args.repo, results, args.cohort, threads=args.threads, force=args.force)
    if args.stage == "prepare":
        from . import prepare
        man = prepare.run(ctx, args.network_list)
        print("prepare: %d networks (%d included) -> %s" % (len(man), (man["status"] == "included").sum(),
                                                             ctx.cohort_dir))
        return 0
    if args.stage == "aggregate":
        from . import aggregate
        inf = aggregate.run(ctx, n_boot=args.n_boot)
        print(inf.to_string())
        return 0
    failures = 0
    for net in _networks(ctx, args):
        units = [None] if args.stage in ("simulate", "structure", "resolution") else _models(args)
        for m in units:
            t0 = time.time()
            try:
                if args.stage == "simulate":
                    from . import simulate
                    r = simulate.run(ctx, net)
                elif args.stage == "structure":
                    from . import structure
                    r = structure.run(ctx, net, with_decomposition=not args.no_decomposition)
                elif args.stage == "fit":
                    from . import fit_stage
                    r = fit_stage.run(ctx, net, m)
                elif args.stage == "explain":
                    from . import explain
                    r = explain.run(ctx, net, m)
                elif args.stage == "decide":
                    from . import decide
                    r = decide.run(ctx, net, m)
                elif args.stage == "resolution":
                    from . import resolution
                    r = resolution.run(ctx, net)
                state = "skipped (current)" if r is None else r.get("status", "ok")
                print("%s %s %s: %s in %.1f s" % (args.stage, net, m or "", state, time.time() - t0), flush=True)
            except Exception:  # noqa: BLE001
                failures += 1
                print("%s %s %s: FAILED\n%s" % (args.stage, net, m or "", traceback.format_exc()), file=sys.stderr,
                      flush=True)
    return 1 if failures else 0
