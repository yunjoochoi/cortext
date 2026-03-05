#!/bin/bash
#SBATCH --job-name=zimg_inpaint
#SBATCH --partition=gigabyte_a6000
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/zimg_inpaint_%j.log

eval "$(conda shell.bash hook)" && conda activate cortext
cd ~/cortext

export HF_HUB_OFFLINE=1

accelerate launch --num_processes 2 training/train_zimage_inpaint_lora.py \
    --pretrained_model_name_or_path /scratch2/shaush/models/models--Tongyi-MAI--Z-Image/snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021 \
    --manifest_path /scratch2/shaush/coreset_output/manifest.jsonl \
    --caption_cache_path /scratch2/shaush/coreset_output/caption_cache.json \
    --output_dir /scratch2/shaush/training_output/zimage_inpaint_lora \
    --resolution 1024 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --max_train_steps 5000 \
    --learning_rate 1e-4 \
    --patch_embed_lr 1e-4 \
    --lr_scheduler cosine \
    --lr_warmup_steps 200 \
    --rank 16 \
    --seed 42 \
    --mixed_precision bf16 \
    --checkpointing_steps 500 \
    --checkpoints_total_limit 3 \
    --report_to tensorboard \
    --dataloader_num_workers 4 \
    --max_sequence_length 256 \
    --mask_loss_only
