#!/bin/bash
#SBATCH --job-name=verify_dup
#SBATCH --partition=gigabyte_a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=logs/verify_dup_%j.log

eval "$(conda shell.bash hook)" && conda activate cortext
cd ~/cortext

python annotation/verify_duplicates.py \
    --manifest /scratch2/shaush/coreset_output/manifest.jsonl \
    --output /scratch2/shaush/coreset_output/manifest_verified.jsonl
