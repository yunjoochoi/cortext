#!/bin/bash
#SBATCH --job-name=qwen_edit
#SBATCH --partition=gigabyte_a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/qwen_edit_%j.log

eval "$(conda shell.bash hook)" && conda activate cortext
cd ~/cortext

python training/test_qwen_edit.py
