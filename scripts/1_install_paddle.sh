#!/bin/bash
#SBATCH --job-name=install_paddle
#SBATCH -p dell_cpu
#SBATCH --qos=cpu_qos 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --output=logs/install_paddle_%j.out

source /home/shaush/miniconda3/etc/profile.d/conda.sh
conda activate cortext

python3 -m pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu130/
python3 -m pip install paddleocr

python3 -c "import paddle; print('paddle:', paddle.__version__)"