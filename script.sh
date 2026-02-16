#!/bin/bash
#SBATCH --job-name=aihub_105
#SBATCH -p dell_cpu
#SBATCH --qos=cpu_qos
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=aihub_105.out
#SBATCH --time=3-00:00:00 

# ...template...