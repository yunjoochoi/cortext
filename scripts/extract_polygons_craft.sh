#!/bin/bash
#SBATCH --job-name=poly_craft
#SBATCH --partition=gigabyte_a6000
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/extract_poly_craft_%j_%a.log
#SBATCH --array=0-1

eval "$(conda shell.bash hook)" && conda activate cortext
cd ~/cortext

WORLD_SIZE=2

python annotation/extract_polygons_craft.py \
    --manifest /scratch2/shaush/coreset_output/manifest.jsonl \
    --output /scratch2/shaush/coreset_output/polygon_refined_craft.jsonl \
    --rank ${SLURM_ARRAY_TASK_ID} \
    --world_size ${WORLD_SIZE}

# After all shards finish, merge with:
# cat /scratch2/shaush/coreset_output/polygon_refined_craft.shard*.jsonl \
#     > /scratch2/shaush/coreset_output/polygon_refined_craft.jsonl
