#!/bin/bash
#SBATCH --job-name=phase4
#SBATCH --output=/home/hklee2/cortext/logs/phase4_%j.log
#SBATCH --error=/home/hklee2/cortext/logs/phase4_%j.err
#SBATCH --partition=dell_cpu
#SBATCH --qos=cpu_qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00

echo "=========================================="
echo "Phase 4: Pending review visualization"
echo "Job Start: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "=========================================="

START_TIME=$SECONDS

source /home/hklee2/cortext/.venv/bin/activate
cd ~/cortext

mkdir -p logs results

python -u dataset/filtering/phase4_pending_review.py \
    --phase3 results/phase3_vlm_verify_merged.jsonl \
    --output_dir /scratch2/hklee2/pending

EXIT_CODE=$?

echo ""
echo "=========================================="
ELAPSED=$(( SECONDS - START_TIME ))
echo "Finished: $(date)"
echo "Total Time: $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"
echo "Exit Code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
