#!/bin/bash
#SBATCH --job-name=build_manifest
#SBATCH -p dell_cpu
#SBATCH --qos=cpu_qos 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --output=logs/build_manifest%j.out

source /home/shaush/miniconda3/etc/profile.d/conda.sh
conda activate cortext

python /home/shaush/cortext/dataset/selection/prepare/build_manifest.py