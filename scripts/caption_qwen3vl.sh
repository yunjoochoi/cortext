#!/bin/bash
#SBATCH --job-name=caption_qwen3vl
#SBATCH --partition=gigabyte_a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=logs/caption_qwen3vl_%j_%a.log
#SBATCH --array=0-3

eval "$(conda shell.bash hook)" && conda activate cortext
cd ~/cortext

WORLD_SIZE=4

python annotation/caption_qwen3vl.py \
    --manifest /scratch2/shaush/coreset_output/manifest.jsonl \
    --output /scratch2/shaush/coreset_output/manifest_captioned.jsonl \
    --model /scratch2/shaush/models/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
    --batch_size 4 \
    --rank ${SLURM_ARRAY_TASK_ID} \
    --world_size ${WORLD_SIZE}

# After all shards finish, run merge manually:
# python annotation/caption_qwen3vl.py \
#     --manifest /scratch2/shaush/coreset_output/manifest.jsonl \
#     --output /scratch2/shaush/coreset_output/manifest_captioned.jsonl \
#     --world_size 4 \
#     --merge_only
