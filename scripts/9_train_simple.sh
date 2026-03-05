#!/bin/bash
#SBATCH --job-name=train_simple
#SBATCH --partition=gigabyte_a6000
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_simple_%j.log

eval "$(conda shell.bash hook)" && conda activate cortext
cd ~/cortext

accelerate launch --num_processes 2 training/train_lora_z_image_simple.py \
    --pretrained_model_name_or_path /scratch2/shaush/models/models--Tongyi-MAI--Z-Image/snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021 \
    --manifest /scratch2/shaush/coreset_output/manifest.jsonl \
    --output_dir /scratch2/shaush/training_output/lora_simple \
    --height 768 --width 1024 \
    --train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --max_train_steps 5000 \
    --learning_rate 1e-4 \
    --rank 16 \
    --mixed_precision bf16 \
    --gradient_checkpointing \
    --checkpointing_steps 500
