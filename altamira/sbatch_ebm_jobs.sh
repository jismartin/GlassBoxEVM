#!/bin/bash
find . -type f -name 'slurm_model_selection_ebm*' | while read -r file; do
    sbatch "$file"
done
