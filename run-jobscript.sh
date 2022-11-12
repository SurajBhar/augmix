#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=h_convnext_pt
#SBATCH --output=h_convnext_pt.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=suraj.bhardwaj@student.uni-siegen.de
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --mem=10GB
#SBATCH --gres=gpu:1

#module load GpuModules
#module load pytorch-py37-cuda11.2-gcc8/1.9.1
#python cifar.py -m resnet18 -op AdamW -sc CosineAnnealingLR
#python cifar.py -m resnet18 -pt -op AdamW -sc CosineAnnealingLR
python cifar.py -m convnext_tiny -pt -op AdamW -sc CosineAnnealingLR
#python cifar.py -m convnext_tiny -op AdamW -sc CosineAnnealingLR