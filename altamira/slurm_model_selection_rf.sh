#!/bin/bash
#SBATCH --job-name=model_selection_rf
#SBATCH --output=model_selection_rf.out
#SBATCH --error=model_selection_rf.err
#SBATCH --time=5:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

module purge
module load PYTHON
srun -n 1 --ntasks-per-node=16 python model_selection.py rf