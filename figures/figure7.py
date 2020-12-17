import matplotlib.pyplot as plt
import numpy as np
from utils import get_data_for_runs

runs = {
    "exp/cifar_online_2500step_shrink0.0_perturb0.01/": "$\lambda = 0.0$",
    "exp/cifar_online_2500step_shrink0.2_perturb0.01/": "$\lambda = 0.2$",
    "exp/cifar_online_2500step_shrink0.4_perturb0.01/": "$\lambda = 0.4$",
    "exp/cifar_online_2500step_shrink0.6_perturb0.01/": "$\lambda = 0.6$",
    "exp/cifar_online_2500step_shrink0.8_perturb0.01/": "$\lambda = 0.8$",
    "exp/cifar_online_2500step_shrink1.0_perturb0.01/": "$\lambda = 1.0$",
}

datas, summaries = get_data_for_runs(runs)
tags = ['online_test_accuracy', 'online_train_time']
nrows = 1
ncols = len(tags)
plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.5))

for idx, tag in enumerate(tags):
    plt.subplot(nrows, ncols, idx + 1)
    for i, title in enumerate(summaries):
        x = np.arange(0, len(datas[tag][i])) * 2.5
        y = datas[tag][i]
        if 'accuracy' in tag:
            y = y * 100
        else:
            y = y / 60
            print(title, sum(y))
        plt.plot(x, y, label=title)

    if idx == 0:
        plt.legend()
    plt.ylabel(tag.replace('_', ' ').replace('online', '').strip().capitalize())
    plt.xlabel('Number of Samples (thousands)')
plt.tight_layout()
plt.savefig('figure7.pdf')