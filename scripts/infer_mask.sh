#!/bin/bash
#SBATCH --job-name=infer_mask
#SBATCH --partition=gigabyte_a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/infer_mask_%j.log

eval "$(conda shell.bash hook)" && conda activate cortext
cd ~/cortext

python training/infer_lora_z_image_simple.py \
    --model_path /scratch2/shaush/models/models--Tongyi-MAI--Z-Image/snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021 \
    --training_dir /scratch2/shaush/training_output/lora_simple_masked \
    --output_dir /scratch2/shaush/training_output/lora_simple_masked/infer_results \
    --prompt \
        "A building with a blue sign, with text '카페라떼' written on it" \
        "A restaurant storefront, with text '닭볶음탕' written on it" \
        "A street sign at night, with text '안녕하세요' written on it" \
        "A shop entrance, with text '커피숍' written on it" \
    --height 880 --width 1184 \
    --num_inference_steps 50 \
    --seed 42
