#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=augmixtest
#SBATCH --output=cifartest.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=suraj.bhardwaj@student.uni-siegen.de
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --mem=10GB
#SBATCH --gres=gpu:1

eval "$(venv shell.bash hook)"
bin activate myenv1
module load GpuModules
module load pytorch-py37-cuda11.2-gcc8/1.9.1
python cifar.py
