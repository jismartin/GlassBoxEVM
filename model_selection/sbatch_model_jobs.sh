#!/bin/bash
read -p "Enter model name: " model_name
pattern="slurm_lhs_model_selection_${model_name}*"
find "./jobs" -maxdepth 1 -type f -name "$pattern" | while read -r file; do
    sbatch "$file"
done
