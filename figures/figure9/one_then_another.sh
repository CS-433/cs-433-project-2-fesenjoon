#!/bin/bash

dataset1=$1
dataset2=$2
dataset3=$3

num_classes=10

if [ $dataset1 == "cifar100" ]; then
	num_classes=100
fi

#for dataset_portion in 0.2 0.4 0.6 0.8 1.0;
for dataset_portion in 0.1 0.3 0.6 1.0;
do
	python3 train.py \
		${dataset1}_${dataset_portion}_data \
		--exp-dir exp/${dataset1}_${dataset_portion}_data \
		--dataset ${dataset1} \
		--lr 0.001 \
		--optimizer sgd \
		--dataset-portion $dataset_portion

	python3 train.py \
		${dataset1}_then_${dataset2}_${dataset_portion}_data \
		--exp-dir exp/${dataset1}_then_${dataset2}_${dataset_portion}_data \
		--dataset ${dataset2} \
		--optimizer sgd \
		--lr 0.001 \
		--checkpoint-num-classes $num_classes \
		--checkpoint exp/${dataset1}_${dataset_portion}_data/final.pt \
		--dataset-portion $dataset_portion
	
	python3 train.py \
		${dataset1}_then_${dataset2}_${dataset_portion}_data_shrink0.3_perturb_0.0001 \
		--exp-dir exp/${dataset1}_then_${dataset2}_${dataset_portion}_data_shrink0.3_perturb_0.0001 \
		--dataset ${dataset2} \
		--optimizer sgd \
		--lr 0.001 \
		--checkpoint-num-classes $num_classes \
		--checkpoint exp/${dataset1}_${dataset_portion}_data/final.pt \
		--checkpoint-shrink 0.3 \
        --checkpoint-perturb 0.0001 \
		--dataset-portion $dataset_portion

	python3 train.py \
		${dataset1}_then_${dataset3}_${dataset_portion}_data \
		--exp-dir exp/${dataset1}_then_${dataset3}_${dataset_portion}_data \
		--dataset ${dataset3} \
		--optimizer sgd \
		--lr 0.001 \
		--checkpoint-num-classes $num_classes \
		--checkpoint exp/${dataset1}_${dataset_portion}_data/final.pt \
		--dataset-portion $dataset_portion
	
	python3 train.py \
		${dataset1}_then_${dataset3}_${dataset_portion}_data_shrink0.3_perturb_0.0001\
		--exp-dir exp/${dataset1}_then_${dataset3}_${dataset_portion}_data_shrink0.3_perturb_0.0001 \
		--dataset ${dataset3} \
		--optimizer sgd \
		--lr 0.001 \
		--checkpoint-num-classes $num_classes \
		--checkpoint exp/${dataset1}_${dataset_portion}_data/final.pt \
		--checkpoint-shrink 0.3 \
        --checkpoint-perturb 0.0001 \
		--dataset-portion $dataset_portion
done


