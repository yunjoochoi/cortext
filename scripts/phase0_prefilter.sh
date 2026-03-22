#!/bin/bash
#SBATCH --job-name=phase0
#SBATCH --output=/home/hklee2/cortext/logs/phase0_%j.log
#SBATCH --error=/home/hklee2/cortext/logs/phase0_%j.err
#SBATCH --partition=dell_cpu
#SBATCH --qos=cpu_qos
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00

echo "=========================================="
echo "Phase 0: Pre-filtering"
echo "Job Start: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "=========================================="

START_TIME=$SECONDS

source /home/hklee2/cortext/.venv/bin/activate
cd ~/cortext

echo "Python: $(which python)"
python --version

mkdir -p logs results

python -u dataset/filtering/phase0_prefilter.py \
    --data_root /scratch2/shaush/030.야외_실제_촬영_한글_이미지 \
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