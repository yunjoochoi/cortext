#!/bin/bash
#SBATCH --job-name=test_ocr
#SBATCH --partition=gigabyte_a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/test_ocr_contrastive_%j.log

eval "$(conda shell.bash hook)" && conda activate cortext
cd ~/cortext

python tests/test_ppocr_contrastive.py
