#!/bin/bash
dataset=$1
config=$2
for seed in 123
do
  if [ $config = 1 ]; then
    python train.py --beta_mode=C   --dataroot=../datasets/$dataset --lambda_irw_A=0.0 --lambda_irw_B=0.0  --resume_epoch=70
  elif [ $config = 2 ]; then
    python train.py --beta_mode=B --dataroot=../datasets/$dataset --threshold=0.1 --lambda_irw_B=1.0 --lambda_irw_A=0.0 --resume_epoch=80
  elif [ $config = 3 ]; then
    python train.py --beta_mode=B --dataroot=../datasets/$dataset --threshold=0.1 --lambda_irw_B=0.5 --lambda_irw_A=0.0 --resume_epoch=80
  elif [ $config = 4 ]; then
    python train.py --beta_mode=B --dataroot=../datasets/$dataset --threshold=0.1 --lambda_irw_B=0.0 --lambda_irw_A=0.0 --resume_epoch=80
  fi
done
