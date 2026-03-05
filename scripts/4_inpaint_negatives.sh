#!/bin/bash
#SBATCH --job-name=inpaint
#SBATCH --partition=gigabyte_a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/inpaint_%j.log

eval "$(conda shell.bash hook)" && conda activate cortext
cd ~/cortext

OUT=/scratch2/shaush/coreset_output

python -m dataset.inpaint \
    --hard_negatives ${OUT}/hard_negatives.jsonl \
    --output_dir ${OUT}/negative_images \
    --device cuda
