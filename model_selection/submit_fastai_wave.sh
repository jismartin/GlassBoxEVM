#!/bin/bash
# Submit one wave of the FastAI selection re-run: the 10 SLURM arrays (50 tasks each) of one subset.
# Usage (from the repository root on Altamira): bash model_selection/submit_fastai_wave.sh J30|J60|J90|J120
set -euo pipefail
SUBSET=${1:?usage: submit_fastai_wave.sh J30|J60|J90|J120}
MAX_SUBMITTED=2000      # QoS limit of submitted jobs per user (array tasks count individually)
WAVE_TASKS=500
mkdir -p model_selection/slurm_logs model_selection/results_fastai
queued=$(squeue -u "$(id -un)" -h -r | wc -l)
if (( queued + WAVE_TASKS > MAX_SUBMITTED )); then
  echo "Refusing: $queued jobs already in the queue; $WAVE_TASKS more would exceed $MAX_SUBMITTED." >&2
  exit 1
fi
projects=$(awk -F, -v s="$SUBSET" 'NR>1 && $2==s {sub(/\.sm$/,"",$3); print $3}' model_selection/project_selection.csv)
[ -z "$projects" ] && { echo "No selection projects for subset $SUBSET" >&2; exit 1; }
for p in $projects; do
  sbatch --parsable "model_selection/jobs/slurm_lhs_model_selection_fastai_array_${p}.sh" | sed "s/^/$p job /"
done
