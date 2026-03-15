#!/bin/bash
#SBATCH --job-name=infer_cn_glyph
#SBATCH --partition=gigabyte_a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/infer_cn_glyph_%j.log

eval "$(conda shell.bash hook)" && conda activate cortext
cd ~/cortext

python training/infer_z_image_controlnet_glyph.py \
    --model_path /scratch2/shaush/models/models--Tongyi-MAI--Z-Image/snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021 \
    --training_dir /scratch2/shaush/training_output/lora_controlnet_glyph \
    --output_dir /scratch2/shaush/training_output/lora_controlnet_glyph/infer_results \
    --prompt \
        "A building with a blue sign, these texts are written on it: '카페라떼'" \
        "A restaurant storefront, these texts are written on it: '닭볶음탕'" \
        "A street sign at night, these texts are written on it: '안녕하세요'" \
        "A shop entrance, these texts are written on it: '커피숍'" \
    --texts \
        "카페라떼" \
        "닭볶음탕" \
        "안녕하세요" \
        "커피숍" \
    --bboxes \
        "200,400,600,150" \
        "150,350,700,180" \
        "300,300,400,120" \
        "250,400,500,150" \
    --height 880 --width 1184 \
    --num_inference_steps 50 \
    --conditioning_scale 0.75 \
    --seed 42
