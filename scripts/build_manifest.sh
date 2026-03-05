#!/bin/bash
#SBATCH --job-name=manifest
#SBATCH --partition=gigabyte_a6000
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/manifest_%j.log

eval "$(conda shell.bash hook)" && conda activate cortext
cd ~/cortext

python dataset/manifest.py \
    --output /scratch2/shaush/manifest_1.jsonl
