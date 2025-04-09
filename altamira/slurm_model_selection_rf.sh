#!/bin/bash
#SBATCH --job-name=model_selection_rf
#SBATCH --output=model_selection_rf.out
#SBATCH --error=model_selection_rf.err
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --chdir=/home/uc27/uc27001/glassboxevm/
module purge
module load PYTHON
source $HOME/mlenv/bin/activate
srun -n 1 --ntasks-per-node=16 python3 model_selection.py rf