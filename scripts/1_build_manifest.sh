#!/bin/bash
#SBATCH --job-name=manifest
#SBATCH --partition=gigabyte_a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/manifest_%j.log

eval "$(conda shell.bash hook)" && conda activate cortext
cd ~/cortext


python dataset/manifest_jsonl.py \
    --output /scratch2/shaush/coreset_output/manifest.jsonl \
    --category_filter "1.간판/1.가로형간판/가로형간판1"


