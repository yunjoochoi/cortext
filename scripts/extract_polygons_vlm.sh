#!/bin/bash
#SBATCH --job-name=extract_poly_vlm
#SBATCH --partition=gigabyte_a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --output=logs/extract_poly_vlm_%j_%a.log
#SBATCH --array=0-3

eval "$(conda shell.bash hook)" && conda activate cortext
cd ~/cortext

WORLD_SIZE=4

python annotation/extract_polygons_vlm.py \
    --manifest /scratch2/shaush/coreset_output/manifest.jsonl \
    --output /scratch2/shaush/coreset_output/polygon_refined_vlm.jsonl \
    --rank ${SLURM_ARRAY_TASK_ID} \
    --world_size ${WORLD_SIZE}

# After all shards finish, merge with:
# cat /scratch2/shaush/coreset_output/polygon_refined_vlm.shard*.jsonl \
#     > /scratch2/shaush/coreset_output/polygon_refined_vlm.jsonl
