#!/bin/bash
#SBATCH --job-name=phase5
#SBATCH --output=/home/hklee2/cortext/logs/phase5_%j.log
#SBATCH --error=/home/hklee2/cortext/logs/phase5_%j.err
#SBATCH --partition=dell_cpu
#SBATCH --qos=cpu_qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

echo "=========================================="
echo "Phase 5: Build final clean dataset"
echo "Job Start: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "=========================================="

START_TIME=$SECONDS

source /home/hklee2/cortext/.venv/bin/activate
cd ~/cortext

mkdir -p logs results

python -u dataset/filtering/phase5_build_dataset.py \
    --phase3 results/phase3_vlm_verify_merged.jsonl \
    --pending_dir /scratch2/hklee2/pending \
    --output_dir /scratch2/hklee2/clean_dataset

EXIT_CODE=$?

echo ""
echo "=========================================="
ELAPSED=$(( SECONDS - START_TIME ))
echo "Finished: $(date)"
echo "Total Time: $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"
echo "Exit Code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
