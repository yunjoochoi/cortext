#!/bin/bash
#SBATCH --job-name=hardneg
#SBATCH --partition=dell_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/hardneg_%j.log

eval "$(conda shell.bash hook)" && conda activate cortext
cd ~/cortext

OUT=/scratch2/shaush/coreset_output

python -m dataset.hard_negative \
    --scored_manifest ${OUT}/manifest_selected.jsonl \
    --out ${OUT}/hard_negatives.jsonl \
    --coverage_cap 500
