#!/bin/bash
#SBATCH --job-name=tr_cn_glyph
#SBATCH --partition=gigabyte_a6000
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00
#SBATCH --output=logs/train_cn_glyph_%j.log

eval "$(conda shell.bash hook)" && conda activate cortext
cd ~/cortext
export PYTHONPATH="$PWD:$PYTHONPATH"

accelerate launch --num_processes 2 training/train_z_image_controlnet_glyph.py \
    --pretrained_model_name_or_path /scratch2/shaush/models/models--Tongyi-MAI--Z-Image/snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021 \
    --manifest /scratch2/shaush/coreset_output/manifest.jsonl \
    --output_dir /scratch2/shaush/training_output/controlnet_glyph \
    --max_pixels 1048576 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_train_steps 18750 \
    --learning_rate 1e-4 \
    --mixed_precision bf16 \
    --gradient_checkpointing \
    --checkpointing_steps 500 \
    --conditioning_scale 1.0 \
    --control_layers "0,2,4,6,8" \
    --resume_from_checkpoint /scratch2/shaush/training_output/controlnet_glyph/checkpoint-4500
