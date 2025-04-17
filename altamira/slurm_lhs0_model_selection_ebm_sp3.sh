#!/bin/bash
#SBATCH --job-name=lhs0_msel_ebm_3
#SBATCH --output=lhs0_msel_ebm_3.out
#SBATCH --error=lhs0_msel_ebm_3.err
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --chdir=/home/uc27/uc27001/glassboxevm/
module purge
module load PYTHON
source $HOME/mlenv/bin/activate
srun -n 1 --ntasks-per-node=16 python3 model_selection.py ebm 0 3