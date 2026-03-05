#!/bin/bash
#SBATCH --job-name=kd_t2i_infer
#SBATCH --partition=gigabyte_a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=logs/kd_t2i_infer_%j.log

eval "$(conda shell.bash hook)" && conda activate cortext
cd ~/cortext

export HF_HUB_OFFLINE=1

TRAIN_OUT=/scratch2/shaush/training_output/kandinsky_t2i_lora

python training/infer_kandinsky_t2i_lora.py \
    --training_dir ${TRAIN_OUT} \
    --output_dir /home/shaush/cortext/kd_t2i_inference \
    --lora_rank 8 \
    --dtype fp16 \
    --baseline
