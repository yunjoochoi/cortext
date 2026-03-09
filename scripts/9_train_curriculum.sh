#!/bin/bash
#SBATCH --job-name=criculum_simple
#SBATCH --partition=gigabyte_a6000
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_curriculum_%j.log

eval "$(conda shell.bash hook)" && conda activate cortext
cd ~/cortext

accelerate launch --num_processes 2 training/train_lora_z_image_curriculum.py \
    --pretrained_model_name_or_path /scratch2/shaush/models/models--Tongyi-MAI--Z-Image/snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021 \
    --scored_manifest /scratch2/shaush/coreset_output/manifest_scored.jsonl \
    --output_dir /scratch2/shaush/training_output/lora_simple_curriculum \
    --max_pixels 1048576 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --steps_per_phase 1500 \
    --learning_rate 1e-4 \
    --rank 16 \
    --mixed_precision bf16 \
    --gradient_checkpointing \
    --checkpointing_steps 500


# 중간에 죽었을 때 재실행 
# --resume_from_checkpoint latest