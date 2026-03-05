#!/bin/bash
#SBATCH --job-name=kd_inpaint
#SBATCH --partition=gigabyte_a6000
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --output=logs/kd_inpaint_%j.log

eval "$(conda shell.bash hook)" && conda activate cortext
cd ~/cortext

OUT=/scratch2/shaush/coreset_output
TRAIN_OUT=/scratch2/shaush/training_output/kandinsky_inpaint_lora

accelerate launch --num_processes 2 training/train_kandinsky_inpaint_lora.py \
    --manifest_path ${OUT}/manifest_selected.jsonl \
    --output_dir ${TRAIN_OUT} \
    --resolution 1200 \
    --train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --lr_scheduler constant \
    --lr_warmup_steps 500 \
    --rank 16 \
    --max_train_steps 10000 \
    --checkpointing_steps 500 \
    --checkpoints_total_limit 100 \
    --gradient_checkpointing \
    --mixed_precision bf16 \
    --allow_tf32 \
    --dataloader_num_workers 4 \
    --seed 42 \
    --report_to tensorboard
