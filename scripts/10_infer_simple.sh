#!/bin/bash
#SBATCH --job-name=infer_simple
#SBATCH --partition=gigabyte_a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=logs/infer_simple_%j.log

eval "$(conda shell.bash hook)" && conda activate cortext
cd ~/cortext

python training/infer_lora_z_image_simple.py \
    --model_path /scratch2/shaush/models/models--Tongyi-MAI--Z-Image/snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021 \
    --training_dir /scratch2/shaush/training_output/lora_simple \
    --prompt \
        "A Korean signage photo with '카페라떼' written on it." \
        "A photo of a store sign with '출입금지' written on it." \
        "A Korean street photo with '맛있는 치킨' written on it." \
        "A photo of a banner with '세일 50% 할인' written on it." \
    --output_dir /home/shaush/cortext/lora_simple_inference_results \
    --height 768 --width 1024 \
    --num_inference_steps 50 \
    --guidance_scale 5.0 \
    --seed 42
