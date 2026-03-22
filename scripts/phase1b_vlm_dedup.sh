#!/bin/bash
#SBATCH --job-name=phase1b
#SBATCH --output=/home/hklee2/cortext/logs/phase1b_%j.log
#SBATCH --error=/home/hklee2/cortext/logs/phase1b_%j.err
#SBATCH --partition=suma_a6000,gigabyte_a6000,tyan_a6000,suma_rtx4090,base_suma_rtx3090,dell_rtx3090,gigabyte_a5000,asus_a5000
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00

echo "=========================================="
echo "Phase 1b: VLM duplicate verification"
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

python -u dataset/filtering/phase1b_vlm_dedup.py \
    --hash_groups results/phase1a_hash_groups_1171792.jsonl \
    --model_name Qwen/Qwen3-VL-8B-Instruct \
    --max_group_size 10

EXIT_CODE=$?

echo ""
echo "=========================================="
ELAPSED=$(( SECONDS - START_TIME ))
echo "Finished: $(date)"
echo "Total Time: $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"
echo "Exit Code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
