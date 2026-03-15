#!/bin/bash
#SBATCH --job-name=verify_captions
#SBATCH --partition=gigabyte_a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=23:00:00
#SBATCH --output=logs/verify_captions_%j.log

eval "$(conda shell.bash hook)" && conda activate cortext
cd ~/cortext
export PYTHONPATH="$PWD:$PYTHONPATH"

python /home/shaush/cortext/annotation/verify_captions.py \
  --input_dir /scratch2/shaush/coreset_output \
  --batch_size 8 --overwrite