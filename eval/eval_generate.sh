#!/bin/bash
#SBATCH --job-name=eval_gen
#SBATCH --partition=gigabyte_a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/eval_gen_%j.log

eval "$(conda shell.bash hook)" && conda activate cortext
cd ~/cortext
export PYTHONPATH="$PWD:$PYTHONPATH"

MODEL=/scratch2/shaush/models/models--Tongyi-MAI--Z-Image/snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021
EVAL_JSONL=/scratch2/shaush/eval/eval_dataset.jsonl

# Baseline: no LoRA
python eval/generate.py \
    --model_path $MODEL \
    --eval_jsonl $EVAL_JSONL \
    --output_dir /scratch2/shaush/eval/gen_baseline \
    --num_samples 4

# LoRA simple (with LoRA)
python eval/generate.py \
    --model_path $MODEL \
    --lora_path /scratch2/shaush/training_output/lora_simple \
    --eval_jsonl $EVAL_JSONL \
    --output_dir /scratch2/shaush/eval/gen_simple \
    --num_samples 4

# LoRA masked (with LoRA)
python eval/generate.py \
    --model_path $MODEL \
    --lora_path /scratch2/shaush/training_output/lora_masked \
    --eval_jsonl $EVAL_JSONL \
    --output_dir /scratch2/shaush/eval/gen_masked \
    --num_samples 4
