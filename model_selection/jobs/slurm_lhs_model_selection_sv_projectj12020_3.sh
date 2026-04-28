#!/bin/bash
#SBATCH --job-name=lhs_msel_sv_j12020_3
#SBATCH --output=lhs_msel_sv_j12020_3.out
#SBATCH --error=lhs_msel_sv_j12020_3.err
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --chdir=/gpfs/users/res/uc27/uc27001/glassboxevm/
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
module purge
source $HOME/mlenv/bin/activate
srun --cpu-bind=cores python3 ./model_selection/model_selection.py sv j12020_3