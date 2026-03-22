#!/bin/bash
#SBATCH --job-name=phase1a
#SBATCH --output=/home/hklee2/cortext/logs/phase1a_%j.log
#SBATCH --error=/home/hklee2/cortext/logs/phase1a_%j.err
#SBATCH --partition=dell_cpu
#SBATCH --qos=cpu_qos
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00

echo "=========================================="
echo "Phase 1a: Hash-based duplicate detection"
echo "Job Start: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "=========================================="

START_TIME=$SECONDS

source /home/hklee2/cortext/.venv/bin/activate
cd ~/cortext

echo "Python: $(which python)"
python --version

mkdir -p logs results

python -u dataset/filtering/phase1a_dedup.py \
    --valid_pairs results/phase0_valid_pairs_1171653.jsonl \
    --threshold 10 \
    --workers 16

EXIT_CODE=$?

echo ""
echo "=========================================="
ELAPSED=$(( SECONDS - START_TIME ))
echo "Finished: $(date)"
echo "Total Time: $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"
echo "Exit Code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
