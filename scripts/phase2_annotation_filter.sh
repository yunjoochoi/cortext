#!/bin/bash
#SBATCH --job-name=phase2
#SBATCH --output=/home/hklee2/cortext/logs/phase2_%j.log
#SBATCH --error=/home/hklee2/cortext/logs/phase2_%j.err
#SBATCH --partition=dell_cpu
#SBATCH --qos=cpu_qos
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00

echo "=========================================="
echo "Phase 2: Annotation quality filtering"
echo "Job Start: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "=========================================="

START_TIME=$SECONDS

source /home/hklee2/cortext/.venv/bin/activate
cd ~/cortext

echo "Python: $(which python)"
python --version

mkdir -p logs results

python -u dataset/filtering/phase2_annotation_filter.py \
    --valid_pairs results/phase0_valid_pairs_1171653.jsonl \
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
