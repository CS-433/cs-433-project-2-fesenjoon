#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
from utils import get_data_for_runs

runs = {
    "aug_first/": "DA Warm Start",
    "aug_second/": "DA Warm Start",
    "half_cifar/": "Warm Start",
    "cifar_from_half_pretrained/": "Warm Start",
}


datas, summaries = get_data_for_runs(runs)

offsets = {
    "DA Warm Start": 350,
    "Warm Start": 200,
}

tags = ["train_accuracy", "test_accuracy"]
labels = ["train accuracy", "test accuracy"]
names = ["DA Warm Start", "Warm Start"]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

for i in range(2):
    ax = axes[i]
    for j, tag in enumerate(tags):
        x = np.arange(offsets[names[i]] * 2)
        y = datas[tag][i]
        y = y * 100
        ax.plot(x, y, label=labels[j])
    ax.set_ylim(0, 100)
    ax.plot([offsets[names[i]], offsets[names[i]]], ax.get_ylim(), '--', c='black')
    ax.legend()
    ax.set_title(names[i])
    ax.set(xlabel='Training Epoch', ylabel='Accuracy')

plt.savefig("fig_aug.png".format(tag))