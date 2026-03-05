#!/bin/bash
#SBATCH --job-name=poisson_test
#SBATCH --partition=gigabyte_a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/poisson_test_%j.log

eval "$(conda shell.bash hook)" && conda activate cortext
cd ~/cortext

python dataset/inpaint_zimage.py     --hard_negatives /scratch2/shaush/coreset_output/hard_negatives.jsonl     --output_dir /scratch2/shaush/coreset_output/negative_images_zimage