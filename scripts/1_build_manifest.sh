#!/bin/bash
#SBATCH --job-name=manifest
#SBATCH --partition=gigabyte_a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/manifest_%j.log

eval "$(conda shell.bash hook)" && conda activate cortext
cd ~/cortext
export PYTHONPATH="$PWD:$PYTHONPATH"

python dataset/manifest.py \
    --format jsonl \
    --output /scratch2/shaush/coreset_output/manifest.jsonl \
    --caption_dir /scratch2/shaush/coreset_output \
    --category_filter "1.간판/1.가로형간판/가로형간판1"


