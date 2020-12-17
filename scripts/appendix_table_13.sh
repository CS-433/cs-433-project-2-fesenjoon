#!/bin/bash

for weight_decay in 0.1 0.01 0.001;
do

python3 train.py \
	cifar10_${weight_decay}_weight_decay \
	--dataset cifar10 \
	--lr 0.001 \
	--optimizer sgd \
	--weight-decay $weight_decay

python3 train.py \
	half_cifar10_${weight_decay}_weight_decay \
	--exp-dir exp/half_cifar_${weight_decay}_weight_decay \
	--dataset half_cifar10 \
	--lr 0.001 \
	--optimizer sgd \
	--weight-decay $weight_decay

python3 train.py \
	half_cifar10_then_cifar10_${weight_decay}_weight_decay \
	--dataset cifar10 \
	--checkpoint exp/half_cifar_${weight_decay}_weight_decay/final.pt \
	--lr 0.001 \
	--optimizer sgd \
	--weight-decay $weight_decay


done

PYTHONPATH=. python3 figures/appendix-table-13.py