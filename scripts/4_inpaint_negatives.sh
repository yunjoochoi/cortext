#!/bin/bash
#SBATCH --job-name=inpaint
#SBATCH --partition=dell_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/inpaint_%j.log

source activate cortext
cd ~/cortext

OUT=/scratch2/shaush/coreset_output

python -m dataset.inpaint \
    --hard_negatives ${OUT}/hard_negatives.jsonl \
    --output_dir ${OUT}/negative_images \
    --device cuda
