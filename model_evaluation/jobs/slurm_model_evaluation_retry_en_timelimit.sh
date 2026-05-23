#!/bin/bash
#SBATCH --job-name=meval_retry_en_tl
#SBATCH --output=/gpfs/users/res/uc27/uc27001/glassboxevm/model_evaluation/slurm_logs/meval_retry_en_tl_%A_%a.out
#SBATCH --error=/gpfs/users/res/uc27/uc27001/glassboxevm/model_evaluation/slurm_logs/meval_retry_en_tl_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --array=0-21%6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --chdir=/gpfs/users/res/uc27/uc27001/glassboxevm

export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

module purge
source $HOME/mlenv/bin/activate

RETRY_CSV=/gpfs/users/res/uc27/uc27001/glassboxevm/model_evaluation/retry_en_timelimit.csv
PROJECT_CSV=/gpfs/users/res/uc27/uc27001/glassboxevm/model_evaluation/project_evaluation.csv

ROW=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$RETRY_CSV" | tr -d '\r')

PROJECT_FROM_RETRY=$(echo "$ROW" | cut -d, -f2)
MODEL=$(echo "$ROW" | cut -d, -f3)
LHS=$(echo "$ROW" | cut -d, -f4)
PROJECT_INDEX=$(echo "$ROW" | cut -d, -f5)

if [ -z "$MODEL" ] || [ -z "$LHS" ] || [ -z "$PROJECT_INDEX" ]; then
    echo "ERROR: could not parse row for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}: ${ROW}"
    exit 1
fi

PROJECT_NAME=$(awk -F, -v id="$PROJECT_INDEX" 'NR > 1 && $1 == id {print $3; exit}' "$PROJECT_CSV" | tr -d '\r' | sed 's/\.sm$//')

if [ -z "$PROJECT_NAME" ]; then
    echo "ERROR: project index=${PROJECT_INDEX} not found in ${PROJECT_CSV}"
    exit 1
fi

if [ -n "$PROJECT_FROM_RETRY" ] && [ "$PROJECT_FROM_RETRY" != "$PROJECT_NAME" ]; then
    echo "ERROR: project mismatch for retry row ${SLURM_ARRAY_TASK_ID}: retry_csv=${PROJECT_FROM_RETRY}, project_evaluation=${PROJECT_NAME}"
    exit 1
fi

echo "Retry EN timelimit task ${SLURM_ARRAY_TASK_ID}: model=${MODEL}, lhs=${LHS}, project_index=${PROJECT_INDEX}, project=${PROJECT_NAME}"

srun --ntasks=1 \
     --cpus-per-task=${SLURM_CPUS_PER_TASK} \
     --cpu-bind=cores \
     python3 ./model_evaluation/model_evaluation.py "$MODEL" "$LHS" "$PROJECT_NAME"
