#!/bin/bash
echo "Train aux model for CIFAR10" 
python ./kd_from_teacher_to_aux.py --dataset cifar10  > ./logs/train_aux_cifar10.txt
