#!/bin/bash
#SBATCH --job-name=test_enc_sim
#SBATCH --partition=gigabyte_a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --output=logs/test_text_encoder_sim_%j.log

eval "$(conda shell.bash hook)" && conda activate cortext
cd ~/cortext

python tests/test_text_encoder_sim.py
