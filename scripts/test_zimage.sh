#!/bin/bash
#SBATCH --job-name=test_zimage
#SBATCH --partition=gigabyte_a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=logs/test_zimage_%j.log

eval "$(conda shell.bash hook)" && conda activate cortext
cd ~/cortext

python inference/test_zimage.py
