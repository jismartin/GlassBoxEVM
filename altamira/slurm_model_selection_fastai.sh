#!/bin/bash
#SBATCH --job-name=model_selection_fastai
#SBATCH --output=model_selection_fastai.out
#SBATCH --error=model_selection_fastai.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --chdir=/home/uc27/uc27001/glassboxevm/
module purge
module load PYTHON
source $HOME/mlenv/bin/activate
srun -n 1 --ntasks-per-node=16 python3 fastai_model_selection.py