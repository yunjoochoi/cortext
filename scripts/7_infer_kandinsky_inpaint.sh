#!/bin/bash
#SBATCH --job-name=kd_infer
#SBATCH --partition=gigabyte_a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=logs/kd_infer_%j.log

eval "$(conda shell.bash hook)" && conda activate cortext
cd ~/cortext

export HF_HUB_OFFLINE=1

TRAIN_OUT=/scratch2/shaush/training_output/kandinsky_inpaint_lora

python cortext/inference/test_zimage.py
