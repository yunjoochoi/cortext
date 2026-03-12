#!/bin/bash
#SBATCH --job-name=test_contra
#SBATCH --partition=gigabyte_a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/test_contrastive_pair_%j.log

eval "$(conda shell.bash hook)" && conda activate cortext
cd ~/cortext

python tests/test_contrastive_pair.py
