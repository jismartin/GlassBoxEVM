#!/bin/bash
#SBATCH --job-name=msel_fastai_j904_1
#SBATCH --output=/gpfs/users/res/uc27/uc27001/glassboxevm/model_selection/slurm_logs/msel_fastai_j904_1_%A_%a.out
#SBATCH --error=/gpfs/users/res/uc27/uc27001/glassboxevm/model_selection/slurm_logs/msel_fastai_j904_1_%A_%a.err
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --array=0-49%25
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --chdir=/gpfs/users/res/uc27/uc27001/glassboxevm

export HOME=${HOME:-/gpfs/users/res/uc27/uc27001}
export MPLBACKEND=Agg
export CACHE_ROOT=/gpfs/users/res/uc27/uc27001/.cache
export XDG_CACHE_HOME=${CACHE_ROOT}
export FASTAI_HOME=${CACHE_ROOT}/fastai
export TORCH_HOME=${CACHE_ROOT}/torch
export MPLCONFIGDIR=${CACHE_ROOT}/matplotlib
mkdir -p ${FASTAI_HOME} ${TORCH_HOME} ${MPLCONFIGDIR}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

type module &>/dev/null && module purge
source /gpfs/users/res/uc27/uc27001/mlenv/bin/activate

srun --ntasks=1 --cpus-per-task=${SLURM_CPUS_PER_TASK} --cpu-bind=cores python3 ./model_selection/model_selection.py fastai j904_1 --lhs ${SLURM_ARRAY_TASK_ID} --path-results ./model_selection/results_fastai
