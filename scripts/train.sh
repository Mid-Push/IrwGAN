#!/bin/bash
dataset=$1
config=$2
for seed in 123
do
  if [ $config = 1 ]; then
    python train.py --beta_mode=C   --dataroot=../datasets/$dataset --lambda_irw_A=0.0 --lambda_irw_B=0.0 
  elif [ $config = 2 ]; then
    python train.py --beta_mode=AB --dataroot=../datasets/$dataset --threshold=0.1 
  elif [ $config = 3 ]; then
    python train.py --beta_mode=AB --dataroot=../datasets/$dataset --threshold=0.0 
  fi
done
