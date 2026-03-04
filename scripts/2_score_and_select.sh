#!/bin/bash
#SBATCH --job-name=coreset
#SBATCH --partition=dell_rtx3090
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/coreset_%j.log

source activate cortext
cd ~/cortext

OUT=/scratch2/shaush/coreset_output

# Score manifest with type-based difficulty tiers
python -m core.difficulty --manifest ${OUT}/manifest.jsonl --out ${OUT}/manifest_scored.jsonl

# Coreset selection
python -m selection.coreset --manifest ${OUT}/manifest_scored.jsonl --output ${OUT}/selected_indices.json

# Export selected coreset
python -m selection.export --manifest ${OUT}/manifest_scored.jsonl --indices ${OUT}/selected_indices.json --output ${OUT}/manifest_selected.jsonl
