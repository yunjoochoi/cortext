#!/bin/bash
#SBATCH --job-name=extract_embeds
#SBATCH -p dell_rtx3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=logs/extract_embeds_%j.out

source /home/shaush/miniconda3/etc/profile.d/conda.sh
conda activate cortext

python /home/shaush/cortext/dataset/selection/run_pipeline.py