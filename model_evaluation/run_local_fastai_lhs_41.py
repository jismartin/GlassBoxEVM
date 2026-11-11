#!/usr/bin/env python
"""Run a FastAI LHS model evaluation locally over project_evaluation.csv.

The LHS hyperparameterization is chosen with --lhs (default: 41, the original
local run; use --lhs 10 for the corrected selection).

This is the local-desktop counterpart of
model_evaluation/jobs/slurm_model_evaluation_fastai_lhs_41.sh.  It launches one
project evaluation per worker process and constrains each process to a small
number of numerical-library threads.  With the defaults, it runs up to 5
projects concurrently and gives each project 1 CPU thread.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROJECTS_CSV = REPO_ROOT / "model_evaluation" / "project_evaluation.csv"
DEFAULT_RESULTS_DIR = REPO_ROOT / "model_evaluation" / "results"
DEFAULT_HYPERPARAMETERS_DIR = REPO_ROOT / "model_evaluation" / "hyperparameters"
DEFAULT_MONTECARLO_DIR = REPO_ROOT / "montecarlo"
DEFAULT_LOG_DIR = REPO_ROOT / "model_evaluation" / "local_logs"
MODEL_EVALUATION_SCRIPT = REPO_ROOT / "model_evaluation" / "model_evaluation.py"


@dataclass(frozen=True)
class Project:
    position: int
    subset: str
    file_name: str

    @property
    def name(self) -> str:
        return self.file_name.removesuffix(".sm")


@dataclass(frozen=True)
class RunResult:
    project: Project
    returncode: int
    elapsed_seconds: float
    stdout_path: Path
    stderr_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run local FastAI model evaluation for one LHS hyperparameterization "
            "over projects listed in model_evaluation/project_evaluation.csv."
        )
    )
    parser.add_argument(
        "--lhs",
        type=int,
        default=41,
        help="LHS hyperparameterization id from lhs_fastai_parameters.csv. Default: 41.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of project evaluations to run concurrently. Default: 5.",
    )
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        default=1,
        help="Numerical-library threads available to each worker process. Default: 1.",
    )
    parser.add_argument(
        "--projects-csv",
        type=Path,
        default=DEFAULT_PROJECTS_CSV,
        help="CSV containing the project list. Default: model_evaluation/project_evaluation.csv.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory where model_evaluation.py writes result CSVs.",
    )
    parser.add_argument(
        "--hyperparameters-dir",
        type=Path,
        default=DEFAULT_HYPERPARAMETERS_DIR,
        help="Directory containing lhs_fastai_parameters.csv.",
    )
    parser.add_argument(
        "--montecarlo-dir",
        type=Path,
        default=DEFAULT_MONTECARLO_DIR,
        help="Directory containing simulation CSVs and test_runs.csv.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help="Directory for per-project stdout/stderr logs.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Zero-based inclusive project position in the CSV to start from.",
    )
    parser.add_argument(
        "--stop",
        type=int,
        default=None,
        help="Zero-based exclusive project position in the CSV to stop at.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of projects to run after slicing/filtering.",
    )
    parser.add_argument(
        "--project",
        action="append",
        default=[],
        help=(
            "Project name to run, with or without .sm. Can be repeated. "
            "When omitted, all projects in the selected slice are run."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip projects whose expected result CSV already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected projects without launching training.",
    )
    return parser.parse_args()


def read_projects(projects_csv: Path) -> list[Project]:
    with projects_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        projects = []
        for position, row in enumerate(reader):
            file_name = (row.get("FileName") or "").strip()
            if not file_name:
                raise ValueError(f"Missing FileName at CSV row position {position}.")
            projects.append(
                Project(
                    position=position,
                    subset=(row.get("SubSet") or "").strip(),
                    file_name=file_name,
                )
            )
    return projects


def expected_result_path(results_dir: Path, project: Project, lhs: int) -> Path:
    return (
        results_dir
        / f"model_evaluation_fastai_lhs_{lhs}_subset_{project.subset}_project_{project.name}.csv"
    )


def select_projects(projects: list[Project], args: argparse.Namespace) -> list[Project]:
    selected = projects[args.start : args.stop]
    if args.project:
        requested = {project.removesuffix(".sm").lower() for project in args.project}
        selected = [project for project in selected if project.name.lower() in requested]
        found = {project.name.lower() for project in selected}
        missing = sorted(requested - found)
        if missing:
            raise ValueError(f"Requested projects not found in selected CSV slice: {missing}")
    if args.skip_existing:
        selected = [
            project
            for project in selected
            if not expected_result_path(args.results_dir, project, args.lhs).exists()
        ]
    if args.limit is not None:
        selected = selected[: args.limit]
    return selected


def worker_env(threads_per_worker: int) -> dict[str, str]:
    env = os.environ.copy()
    thread_value = str(threads_per_worker)
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMBA_NUM_THREADS",
        "SLURM_CPUS_PER_TASK",
    ):
        env[name] = thread_value
    env["PYTHONUNBUFFERED"] = "1"
    return env


def run_project(project: Project, args: argparse.Namespace, env: dict[str, str]) -> RunResult:
    args.log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = args.log_dir / f"fastai_lhs_{args.lhs}_{project.name}.out"
    stderr_path = args.log_dir / f"fastai_lhs_{args.lhs}_{project.name}.err"

    command = [
        sys.executable,
        str(MODEL_EVALUATION_SCRIPT),
        "fastai",
        str(args.lhs),
        project.name,
        "--path-results",
        str(args.results_dir),
        "--path-hyperparameters",
        str(args.hyperparameters_dir),
        "--path-montecarlo",
        str(args.montecarlo_dir),
        "--path-project-evaluation",
        str(args.projects_csv),
    ]

    started = time.time()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        process = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=stdout,
            stderr=stderr,
            text=True,
            check=False,
        )

    return RunResult(
        project=project,
        returncode=process.returncode,
        elapsed_seconds=time.time() - started,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.threads_per_worker < 1:
        raise ValueError("--threads-per-worker must be >= 1")

    args.projects_csv = args.projects_csv.resolve()
    args.results_dir = args.results_dir.resolve()
    args.hyperparameters_dir = args.hyperparameters_dir.resolve()
    args.montecarlo_dir = args.montecarlo_dir.resolve()
    args.log_dir = args.log_dir.resolve()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    projects = select_projects(read_projects(args.projects_csv), args)
    print(
        f"Selected {len(projects)} projects. "
        f"Running with {args.workers} workers x {args.threads_per_worker} thread(s)."
    )

    if args.dry_run:
        for project in projects:
            print(f"{project.position}: {project.name} ({project.subset})")
        return 0

    if not projects:
        print("Nothing to run.")
        return 0

    env = worker_env(args.threads_per_worker)
    failures: list[RunResult] = []
    started_all = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_project, project, args, env): project for project in projects
        }
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            status = "OK" if result.returncode == 0 else "FAILED"
            elapsed = round(result.elapsed_seconds, 1)
            print(
                f"[{index}/{len(projects)}] {status} {result.project.name} "
                f"in {elapsed}s"
            )
            if result.returncode != 0:
                failures.append(result)
                print(f"  stderr: {result.stderr_path}")

    total_elapsed = round(time.time() - started_all, 1)
    if failures:
        print(f"Finished with {len(failures)} failure(s) after {total_elapsed}s.")
        for failure in failures:
            print(f"  {failure.project.name}: {failure.stderr_path}")
        return 1

    print(f"Finished all {len(projects)} project(s) after {total_elapsed}s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
