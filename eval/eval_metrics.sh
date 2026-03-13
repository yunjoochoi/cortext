#!/bin/bash
#SBATCH --job-name=eval_metrics
#SBATCH --partition=gigabyte_a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --output=logs/eval_metrics_%j.log

eval "$(conda shell.bash hook)" && conda activate cortext
cd ~/cortext
export PYTHONPATH="$PWD:$PYTHONPATH"

EVAL_JSONL=/scratch2/shaush/eval/eval_dataset.jsonl
GT_IMG_DIR=/scratch2/shaush/eval/gt_images

for METHOD in baseline simple masked; do
    GEN_DIR=/scratch2/shaush/eval/gen_${METHOD}
    echo "=== ${METHOD} ==="

    echo "--- OCR Accuracy ---"
    python eval/eval_ocr.py \
        --img_dir $GEN_DIR \
        --eval_jsonl $EVAL_JSONL \
        --num_samples 4 \
        --lang korean \
        --output $GEN_DIR/ocr_results.jsonl

    echo "--- CLIPScore ---"
    python eval/eval_clipscore.py \
        --img_dir $GEN_DIR \
        --eval_jsonl $EVAL_JSONL \
        --num_samples 4

    echo "--- FID ---"
    python -m pytorch_fid $GT_IMG_DIR $GEN_DIR

    echo ""
done
