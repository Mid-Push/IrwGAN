#!/bin/bash
#SBATCH --partition=GPU-AI
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta16:1
#SBATCH --time=48:00:00
#SBATCH --output=./logs/out.log
#SBATCH --error=./logs/error.log
conda activate project
cd /pylon5/ac5616p/yanwuxu/gan/IRW_GAN
sh scripts/train.sh $1 $2

