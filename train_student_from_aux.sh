#!/bin/bash
# echo "Train student model for CIFAR10 - CPKD-CLS" 
python ./kd_from_aux_to_student.py --dataset cifar10 --inter_method indistill --single_layer_method pkt --scheme cls > ./logs/train_student_from_aux_indistill_cls_cifar10.txt

