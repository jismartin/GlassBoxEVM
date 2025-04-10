#!/bin/bash
#SBATCH --job-name=model_selection_ebm_1_0_0.04
#SBATCH --output=model_selection_ebm_1_0_0.04.out
#SBATCH --error=model_selection_ebm_1_0_0.04.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --chdir=/home/uc27/uc27001/glassboxevm/
module purge
module load PYTHON
source $HOME/mlenv/bin/activate
srun -n 1 --ntasks-per-node=16 python3 ebm_model_selection.py 1 0 0.04