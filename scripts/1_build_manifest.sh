#!/bin/bash
#SBATCH --job-name=manifest
#SBATCH --partition=dell_rtx3090
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/manifest_%j.log

source activate cortext
cd ~/cortext

python -m dataset.manifest configs/config.yaml
