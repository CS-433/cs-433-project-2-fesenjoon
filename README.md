

# Reproducibility Challenge: On Warm-Starting Neural Network Training
This repository contains the code used to reproduce and validate the results for the paper On Warm-Starting Neural Network Training: [https://arxiv.org/abs/1910.08475](https://arxiv.org/abs/1910.08475)


##

## Installation

The implementation is in Python 3.8. You may use the following command to install the necessary packages.

```pip install -r requirements.txt```


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
   * `data_augmentation_gradients.sh`: Generates Figure 8 of the original paper with data augmentation instead of shirnk perturb.

## Contributors
* Klim Kireev
* Amirkeivan Mohtashami
* Ehsan Pajouheshgar

This challenge was completed as the second project of CS-433 class at EPFL.