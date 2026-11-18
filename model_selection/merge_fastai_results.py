#!/usr/bin/env python
"""Merge the FastAI selection re-run (one CSV per project and LHS) into one CSV per project.

Reads  model_selection/results_fastai/lhs_model_selection_fastai_lhs_<k>_project_<p>.csv
Writes model_selection/results/lhs_model_selection_fastai_project_<p>.csv  (only complete projects,
       same columns as the other families, rows sorted by LHS)
       model_selection/fastai_missing.csv  (project,lhs pairs without a result)
       model_selection/jobs/slurm_lhs_model_selection_fastai_retry.sh  (array over the missing pairs)
Run from the repository root.
"""
import argparse
import glob
import os
import re

import pandas as pd

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--time", default="00:30:00", help="Time limit of the retry array tasks.")
parser.add_argument("--max-concurrent", type=int, default=25)
args = parser.parse_args()

CLUSTER_ROOT = "/gpfs/users/res/uc27/uc27001/glassboxevm"
projects = [f.replace(".sm", "") for f in pd.read_csv("model_selection/project_selection.csv")["FileName"]]
n_lhs = len(pd.read_csv("model_selection/hyperparameters/lhs_fastai_parameters.csv"))

found = {}
pattern = re.compile(r"lhs_model_selection_fastai_lhs_(\d+)_project_(.+)\.csv$")
for path in glob.glob("model_selection/results_fastai/lhs_model_selection_fastai_lhs_*_project_*.csv"):
    m = pattern.search(os.path.basename(path))
    if m:
        found[(m.group(2), int(m.group(1)))] = path

missing, complete = [], 0
for p in projects:
    paths = [found.get((p, k)) for k in range(n_lhs)]
    missing += [(p, k) for k, path in enumerate(paths) if path is None]
    if all(paths):
        df = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True).sort_values("lhs")
        assert len(df) == n_lhs and df["lhs"].tolist() == list(range(n_lhs)), p
        df.to_csv(f"model_selection/results/lhs_model_selection_fastai_project_{p}.csv", index=False)
        complete += 1

pd.DataFrame(missing, columns=["project", "lhs"]).to_csv("model_selection/fastai_missing.csv", index=False)
print(f"complete projects: {complete}/{len(projects)} | results: {len(found)}/{len(projects) * n_lhs} | missing: {len(missing)}")

retry = "model_selection/jobs/slurm_lhs_model_selection_fastai_retry.sh"
if missing:
    with open(retry, "w", newline="\n") as f:
        f.write("#!/bin/bash\n")
        f.write("#SBATCH --job-name=msel_fastai_retry\n")
        f.write(f"#SBATCH --output={CLUSTER_ROOT}/model_selection/slurm_logs/msel_fastai_retry_%A_%a.out\n")
        f.write(f"#SBATCH --error={CLUSTER_ROOT}/model_selection/slurm_logs/msel_fastai_retry_%A_%a.err\n")
        f.write(f"#SBATCH --time={args.time}\n#SBATCH --mem=4G\n")
        f.write(f"#SBATCH --array=0-{len(missing) - 1}%{args.max_concurrent}\n")
        f.write("#SBATCH --nodes=1\n#SBATCH --ntasks=1\n#SBATCH --cpus-per-task=1\n#SBATCH --hint=nomultithread\n")
        f.write(f"#SBATCH --chdir={CLUSTER_ROOT}\n\n")
        f.write("export HOME=${HOME:-/gpfs/users/res/uc27/uc27001}\nexport MPLBACKEND=Agg\n")
        f.write("export CACHE_ROOT=/gpfs/users/res/uc27/uc27001/.cache\nexport XDG_CACHE_HOME=${CACHE_ROOT}\n")
        f.write("export FASTAI_HOME=${CACHE_ROOT}/fastai\nexport TORCH_HOME=${CACHE_ROOT}/torch\n")
        f.write("export MPLCONFIGDIR=${CACHE_ROOT}/matplotlib\nmkdir -p ${FASTAI_HOME} ${TORCH_HOME} ${MPLCONFIGDIR}\n")
        for v in ["OPENBLAS", "OMP", "MKL", "NUMEXPR"]:
            f.write(f"export {v}_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}\n")
        f.write("\ntype module &>/dev/null && module purge\n")
        f.write("source /gpfs/users/res/uc27/uc27001/mlenv/bin/activate\n\n")
        f.write('LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" model_selection/fastai_missing.csv)\n')
        f.write("PROJECT=${LINE%%,*}\nLHS=${LINE##*,}\n")
        f.write("srun --ntasks=1 --cpus-per-task=${SLURM_CPUS_PER_TASK} --cpu-bind=cores "
                "python3 ./model_selection/model_selection.py fastai ${PROJECT} --lhs ${LHS} "
                "--path-results ./model_selection/results_fastai\n")
    print(f"retry array written: {retry} ({len(missing)} tasks)")
elif os.path.exists(retry):
    os.remove(retry)
