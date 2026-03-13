#!/bin/bash
#SBATCH --job-name=verify_captions
#SBATCH --partition=gigabyte_a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/verify_captions_%j.log

eval "$(conda shell.bash hook)" && conda activate cortext
cd ~/cortext
export PYTHONPATH="$PWD:$PYTHONPATH"

python annotation/verify_captions.py \
    --input /scratch2/shaush/coreset_output/captions_shard_0.jsonl \
    --batch_size 8
