#!/bin/bash
#SBATCH --job-name=meval_rf_14
#SBATCH --output=/gpfs/users/res/uc27/uc27001/glassboxevm/model_evaluation/slurm_logs/meval_rf_14_%A_%a.out
#SBATCH --error=/gpfs/users/res/uc27/uc27001/glassboxevm/model_evaluation/slurm_logs/meval_rf_14_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --array=0-359%30
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

PROJECT_NAME=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" /gpfs/users/res/uc27/uc27001/glassboxevm/model_evaluation/project_evaluation.csv | cut -d, -f3 | sed 's/\.sm$//')

srun --ntasks=1 --cpus-per-task=${SLURM_CPUS_PER_TASK} --cpu-bind=cores python3 ./model_evaluation/model_evaluation.py rf 14 ${PROJECT_NAME}
