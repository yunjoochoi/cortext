#!/bin/bash
#SBATCH --job-name=extract_poly
#SBATCH --partition=dell_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --output=logs/extract_poly_%j.log

eval "$(conda shell.bash hook)" && conda activate cortext
cd ~/cortext

python annotation/extract_polygons.py \
    --data_root /scratch2/shaush/030.야외_실제_촬영_한글_이미지 \
    --output /scratch2/shaush/coreset_output/polygon_lookup.jsonl
