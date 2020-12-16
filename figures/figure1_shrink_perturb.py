#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
from utils import get_data_for_runs

runs = {
    "cifar/": "Random Initialization",
    "half_cifar/": "Warm-Starting (No Shrink-Perturb)",
    "cifar_from_half_pretrained_shrink0.1_perturb0.01/": "$\lambda = 0.1$",
    "cifar_from_half_pretrained_shrink0.3_perturb0.01/": "$\lambda = 0.3$",
    "cifar_from_half_pretrained_shrink0.6_perturb0.01/": "$\lambda = 0.6$",
}


datas, summaries = get_data_for_runs(runs)

for tag in ["test_accuracy"]:
    plt.figure()
    for i, title in enumerate(summaries):
        x = np.arange(0, len(datas[tag][i]))
        y = datas[tag][i]
        y = y * 100
        plt.plot(x, y, label=title)
    plt.legend()
    
    plt.ylabel(" ".join(tag.split("_")).capitalize())
    plt.ylim(45, 85)
    plt.savefig("fig1_shrink_pertrub_{}.pdf".format(tag))


