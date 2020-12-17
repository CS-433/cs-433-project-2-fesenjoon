

# Reproducibility Challenge: On Warm-Starting Neural Network Training
This repository contains the code used to reproduce and validate the results for the paper On Warm-Starting Neural Network Training: [https://arxiv.org/abs/1910.08475](https://arxiv.org/abs/1910.08475)


##

## Installation

The implementation is in Python 3.8. You may use the following command to install the necessary packages.

```pip install -r requirements.txt```

## Libraries

We use the following libraries:
   
* PyTorch: For all deep learning implementations
* Matplotlib: For plotting
* Torchvision: For loading datasets
* Tensorboard: To store results
* Tensorflow: To read tensorboard format
* Scipy: Used in data processing of SVHN by Torchvision

## Experiments

A separate bash script exists for each of the experiments in the `scripts` folder. 
Run them from the root of project. 
For example, the following command will generate Figure 1 of the original paper.

```
bash scripts/figure1.sh
```


The scripts are named after figures/tables of the original paper. 
The exceptions are the scripts for the additional experiments in the reproducibility report. 
These include:
   * `figure7_offline.sh`: Generates Figure 1 of the original paper with shrink perturb method.
   * `data_augmentation.sh`: Generates Figure 1 of the original paper with data augmentation.
   * `figure5_augmentation.sh`: Generates Figure 5 of the original paper with data augmentation instead of shirnk perturb.

## Pre-trained Models
The weights for ResNet-18 trained on half of CIFAR10 and checkpointed after every 10th epoch is available at: https://github.com/CS-433/cs-433-project-2-fesenjoon/releases/download/resnet18-half-cifar10/Dec12_21-44-54.zip

To use, extract the zip file and put the contents in `exp/half_cifar`. 

Then run the commands normally excluding the one for training a model on half of CIFAR10. 

Furthermore, results of individual runs for Table 1 (without averaging) is available under `tables` directory in NumPy format. 

## Contributors
* Klim Kireev
* Amirkeivan Mohtashami
* Ehsan Pajouheshgar

This challenge was completed as the second project of CS-433 class at EPFL.