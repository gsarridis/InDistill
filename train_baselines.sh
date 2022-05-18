#!/bin/bash
echo "Train baseline models (Teacher, Aux, and Student)... " 
echo "Dataset CIFAR10"
python ./train_baselines.py --dataset cifar10 --ep 50 30  > ./logs/train_baselines_cifar10.txt
