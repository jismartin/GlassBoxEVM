#!/bin/bash
#SBATCH --job-name=meval_retry
#SBATCH --output=/gpfs/users/res/uc27/uc27001/glassboxevm/model_evaluation/slurm_logs/meval_retry_%A_%a.out
#SBATCH --error=/gpfs/users/res/uc27/uc27001/glassboxevm/model_evaluation/slurm_logs/meval_retry_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --array=0-303%6
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

ERROR_CSV=/gpfs/users/res/uc27/uc27001/glassboxevm/model_evaluation/job_error.csv
PROJECT_CSV=/gpfs/users/res/uc27/uc27001/glassboxevm/model_evaluation/project_evaluation.csv

ROW=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$ERROR_CSV" | tr -d '\r')

MODEL=$(echo "$ROW" | cut -d, -f1)
LHS=$(echo "$ROW" | cut -d, -f2)
PROJECT_ID=$(echo "$ROW" | cut -d, -f3)

if [ -z "$MODEL" ] || [ -z "$LHS" ] || [ -z "$PROJECT_ID" ]; then
    echo "ERROR: could not parse row for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}: ${ROW}"
    exit 1
fi

PROJECT_NAME=$(sed -n "$((PROJECT_ID + 2))p" "$PROJECT_CSV" | cut -d, -f3 | sed 's/\.sm$//')

if [ -z "$PROJECT_NAME" ]; then
    echo "ERROR: project_id=${PROJECT_ID} does not map to a project row in ${PROJECT_CSV}"
    exit 1
fi

echo "Retry task ${SLURM_ARRAY_TASK_ID}: model=${MODEL}, lhs=${LHS}, project_id=${PROJECT_ID}, project=${PROJECT_NAME}"

srun --ntasks=1 \
     --cpus-per-task=${SLURM_CPUS_PER_TASK} \
     --cpu-bind=cores \
     python3 ./model_evaluation/model_evaluation.py "$MODEL" "$LHS" "$PROJECT_NAME"
