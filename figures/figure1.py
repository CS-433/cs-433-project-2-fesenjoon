#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
from utils import get_data_for_runs

runs = {
    "half_cifar/": "50%",
    "cifar_from_half_pretrained/": "50%",
    "cifar/": "100%",
}


datas, summaries = get_data_for_runs(runs)

offsets = {
    "100%": 200,
}

for tag in ["train_accuracy", "test_accuracy"]:
    plt.figure()
    for i, title in enumerate(summaries):
        offset = offsets.get(title, 0)
        x = np.arange(offset, len(datas[tag][i]) + offset)
        y = datas[tag][i]
        y = y * 100
        plt.plot(x, y, label=title)
    plt.legend()
    
    plt.ylabel(" ".join(tag.split("_")).capitalize())
    plt.ylim(0, 100)
    plt.plot([200, 200], plt.gca().get_ylim(), '--', c='black')
    plt.savefig("fig1_{}.pdf".format(tag))


