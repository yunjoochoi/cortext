#!/bin/bash
#SBATCH --job-name=phase3
#SBATCH --output=/home/hklee2/cortext/logs/phase3_%j.log
#SBATCH --error=/home/hklee2/cortext/logs/phase3_%j.err
#SBATCH --partition=suma_a6000,gigabyte_a6000,tyan_a6000,suma_rtx4090,base_suma_rtx3090,dell_rtx3090,gigabyte_a5000,asus_a5000
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00

echo "=========================================="
echo "Phase 3: VLM annotation verification"
echo "Job Start: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "=========================================="

START_TIME=$SECONDS

source /home/hklee2/cortext/.venv/bin/activate
cd ~/cortext

echo "Python: $(which python)"
python --version

mkdir -p logs results

python -u dataset/filtering/phase3_vlm_verify.py \
    --phase1b results/phase1b_vlm_dedup_XXXX_1.jsonl results/phase1b_vlm_dedup_XXXX_2.jsonl results/phase1b_vlm_dedup_XXXX_3.jsonl results/phase1b_vlm_dedup_XXXX_4.jsonl \
    --phase2 results/phase2_annotation_filter_1171718.jsonl \
    --model_name Qwen/Qwen3-VL-8B-Instruct

EXIT_CODE=$?

echo ""
echo "=========================================="
ELAPSED=$(( SECONDS - START_TIME ))
echo "Finished: $(date)"
echo "Total Time: $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"
echo "Exit Code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
