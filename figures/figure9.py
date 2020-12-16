import matplotlib.pyplot as plt
import numpy as np
from utils import get_data_for_runs

datasets = ["cifar10", "cifar100", "svhn"]

runs = {}
for dataset1 in datasets:
    for dataset2 in datasets:
        if dataset1 == dataset2:
            continue
        key = "{}-{}".format(dataset2, dataset1)
        runs[key] = {
            "fresh": {
                "exp/{}_{}_data".format(dataset1, y): y for y in [0.3, 0.6, 1.0]
            },
            "warm": {
                "exp/{}_then_{}_{}_data".format(dataset1, dataset2, y): y for y in [0.3, 0.6, 1.0]
            },
            "shrink perturb": {
                "exp/{}_then_{}_{}_data_shrink0.3_perturb_0.0001".format(dataset1, dataset2, y): y for y in [0.3, 0.6, 1.0]
            }
        }

nrows = 2
ncols = (len(runs) + 1) // nrows
plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3))
plt.tight_layout(pad=2.5)
for idx, (experiment, experiment_runs) in enumerate(runs.items()):
    plt.subplot(nrows, ncols, idx + 1)
    final_dataset, initial_dataset = experiment.upper().split('-')
    for cat in ["warm", "fresh", "shrink perturb"]:
        exps = experiment_runs[cat]
        datas, summaries = get_data_for_runs(exps)
        xs = []
        ys = []
        for i, title in enumerate(summaries):
            tag = 'test_accuracy'
            xs.append(title)
            ys.append((datas[tag][i][-1]) * 100)
        plt.plot(xs, ys, label=cat)
    if idx == ncols:
        plt.legend()

    plt.ylabel("Accuracy({})".format(final_dataset))
    plt.title("Pretrained with {}".format(initial_dataset))
    # if "accuracy" in tag:
    # plt.ylim(0, 100)
    # plt.plot([200, 200], plt.gca().get_ylim(), '--', c='black')
    # plt.savefig("fig1_{}.pdf".format(tag))
    plt.savefig("figure9.pdf")