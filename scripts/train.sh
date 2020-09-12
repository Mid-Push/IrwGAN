#!/bin/bash

dataset_config=$1
config=$2
if [ $dataset_config = 1 ]; then
    dataset=vangogh2photo
fi


for seed in 123
do
  if [ $config = 1 ]; then
    # train baseline on noisy dataset
    python train.py --beta_mode=baseline   --dataroot=datasets/$dataset 
  # tuning method
  elif [ $config = 2 ]; then
    python train.py --beta_mode=method --dataroot=datasets/$dataset --lambda_irw_A=0.0 --lambda_irw_B=1.0
  elif [ $config = 3 ]; then
    python train.py --beta_mode=method --dataroot=datasets/$dataset --lambda_irw_A=0.1 --lambda_irw_B=1.0
  elif [ $config = 4 ]; then
    python train.py --beta_mode=method --dataroot=datasets/$dataset --lambda_irw_A=0.5 --lambda_irw_B=1.0
  elif [ $config = 5 ]; then
    python train.py --beta_mode=method --dataroot=datasets/$dataset --lambda_irw_A=1.0 --lambda_irw_B=1.0
  fi
done
