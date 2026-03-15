#!/bin/bash
#SBATCH --job-name=fix_quoted
#SBATCH --partition=gigabyte_a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=23:00:00
#SBATCH --output=logs/fix_quoted_%j.log

eval "$(conda shell.bash hook)" && conda activate cortext
cd ~/cortext
export PYTHONPATH="$PWD:$PYTHONPATH"

python /home/shaush/cortext/annotation/fix_quoted_captions.py \
  --input_dir /scratch2/shaush/coreset_output \
  --batch_size 16
