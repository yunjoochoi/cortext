#!/bin/bash
#SBATCH --job-name=build_curriculum
#SBATCH --partition=dell_rtx3090
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/build_curriculum_%j.log

eval "$(conda shell.bash hook)" && conda activate cortext
cd ~/cortext

echo "=== Step 1: Score manifest and split into curriculum tiers ==="
python -m core.difficulty \
    --manifest /scratch2/shaush/coreset_output/manifest.jsonl \
    --out /scratch2/shaush/coreset_output/manifest_scored.jsonl

echo ""
echo "=== Step 2: Generate hard negatives ==="
python -m dataset.hard_negative \
    --scored_manifest /scratch2/shaush/coreset_output/manifest_scored.jsonl \
    --out /scratch2/shaush/coreset_output/hard_negatives.jsonl

echo ""
echo "=== Done ==="
