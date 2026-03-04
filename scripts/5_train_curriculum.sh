#!/bin/bash
#SBATCH --job-name=curriculum
#SBATCH --partition=dell_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --output=logs/curriculum_%j.log

source activate cortext
cd ~/cortext

OUT=/scratch2/shaush/coreset_output
TRAIN_OUT=/scratch2/shaush/training_output/curriculum_lora

python -m training.curriculum \
    --pretrained_model_name_or_path /scratch2/shaush/models/models--Tongyi-MAI--Z-Image \
    --scored_manifest ${OUT}/manifest_selected.jsonl \
    --hard_negatives_jsonl ${OUT}/negative_images/hard_negatives_with_images.jsonl \
    --output_dir ${TRAIN_OUT} \
    --max_train_steps_per_phase 500 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --rank 16 \
    --mixed_precision bf16 \
    --contrastive_proj_dim 1024
