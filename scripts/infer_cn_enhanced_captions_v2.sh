#!/bin/bash
#SBATCH --job-name=infer_cn_enh_v2
#SBATCH --partition=gigabyte_a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/infer_cn_enhanced_captions_v2_%j.log

eval "$(conda shell.bash hook)" && conda activate cortext
cd ~/cortext

python training/infer_z_image_controlnet_glyph.py \
    --model_path /scratch2/shaush/models/models--Tongyi-MAI--Z-Image/snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021 \
    --training_dir /scratch2/shaush/training_output/controlnet_glyph_enhanced_captions_v2 \
    --output_dir /scratch2/shaush/training_output/controlnet_glyph_enhanced_captions_v2/infer_results \
    --prompt \
        "A close-up of a blue horizontal sign mounted on a beige concrete building under soft afternoon light, with the words of '카페라떼' located at center." \
        "A wide shot of a Korean restaurant storefront with warm yellow lighting and a red awning at dusk, textual material depicted in the image are '닭볶음탕' placed on center." \
        "A nighttime view of a narrow alley with neon-lit street signs reflecting on wet pavement, that reads '안녕하세요' positioned at center." \
        "A daytime photograph of a small shop entrance with a wooden door and potted plants beside a glass window, the written materials on the picture: '커피숍' located in center." \
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
    --guidance_scale 7.5 \
    --num_inference_steps 50 \
    --conditioning_scale 1.0 \
    --seed 42
