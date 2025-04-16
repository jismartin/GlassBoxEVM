#!/bin/bash
read -p "Enter LHS trial number: " trial_number
read -p "Enter model name: " model_name
pattern="slurm_lhs${trial_number}_model_selection_${model_name}*"
find . -type f -name "$pattern" | while read -r file; do
    sbatch "$file"
done

#find . -type f -name 'slurm_lhs0_selection_*' | while read -r file; do
#    sbatch "$file"
#done
