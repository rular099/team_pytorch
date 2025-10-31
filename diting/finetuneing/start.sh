#!/bin/bash

python main.py --seed 0 --mode "train_test" --model-name "seist_l_dis" --log-dir "./logs" --device "cuda:0" --data "D:\GitHub\dataset\LSD" --dataset-name "lsd" --data-split true --train-size 0.8 --val-size 0.1 --shuffle true --workers 8 --in-samples 6000 --augmentation False --epochs 1 --patience 30 --batch-size 16