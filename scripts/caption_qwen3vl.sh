#!/bin/bash
#SBATCH --job-name=caption_qwen3vl
#SBATCH --partition=gigabyte_a6000
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/caption_qwen3vl_%j.log

eval "$(conda shell.bash hook)" && conda activate cortext
cd ~/cortext

python scripts/caption_qwen3vl.py \
    --data-root /scratch2/shaush/030.야외_실제_촬영_한글_이미지 \
    --label-subdir "[라벨]Training" \
    --model-path /scratch2/shaush/models/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
    --cache-path /scratch2/shaush/coreset_output/caption_cache.json \
    --save-every 200
