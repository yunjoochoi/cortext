#!/bin/bash
#SBATCH --job-name=train_attend
#SBATCH --partition=gigabyte_a6000
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_attend_%j.log

eval "$(conda shell.bash hook)" && conda activate cortext
cd ~/cortext

accelerate launch --num_processes 2 training/train_lora_z_image_simple_attend.py \
    --pretrained_model_name_or_path /scratch2/shaush/models/models--Tongyi-MAI--Z-Image/snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021 \
    --manifest /scratch2/shaush/coreset_output/manifest.jsonl \
    --output_dir /scratch2/shaush/training_output/lora_attend \
    --max_pixels 1048576 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_train_steps 5000 \
    --learning_rate 1e-4 \
    --rank 16 \
    --mixed_precision bf16 \
    --gradient_checkpointing \
    --checkpointing_steps 500 \
    --char_loss_lambda 0.05 \
    --char_loss_layers 12,13,14,15,16 \
    --resume_from_checkpoint /scratch2/shaush/training_output/lora_attend/checkpoint-1500