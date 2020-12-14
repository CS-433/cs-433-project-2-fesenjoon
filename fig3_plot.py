#%%
import  argparse
import json
from datetime import datetime
import os
import numpy as np
import time
import torch
import matplotlib.pyplot as plt

def build_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--index_start', type=int, default=0)
    parser.add_argument('--index_stop', type=int, default=100)
    return parser


def main(args):
    bsizes = [16, 32, 64, 128]
    lrs = [0.001, 0.01, 0.1]

    warm_costs = []
    random_costs = []
    warm_accuracy = []
    random_accuracy = []

    for i in range(args.index_start, args.index_stop + 1):
        np.random.seed(i + args.random_seed)
        for bsize in bsizes:
            for lr in lrs:
                for j in range(3):
                    rs = np.random.randint(0, 100000)
                    #print(np.random.randint(0, 3), np.random.randint(0, 4))
                    arr = np.loadtxt('exp/warm/' + str(i) + '_' + str(lr) + '_' + str(bsize) + '_' + str(j) + '.csv')
                    warm_accuracy.append(arr[1])
                    warm_costs.append(arr[2])
                    arr = np.loadtxt('exp/random/' + str(i) + '_' + str(lr) + '_' + str(bsize) + '_' + str(j) + '.csv')
                    random_accuracy.append(arr[1])
                    random_costs.append(arr[2])                    

    print(warm_costs)
    plt.scatter(warm_costs, warm_accuracy, c='b', label='warm start')
    plt.scatter(random_costs, random_accuracy, c='y', label='random start')
    plt.legend()
    plt.ylabel('Test accuracy, %')
    plt.xlabel('Training time, sec')
    plt.savefig('fig3.png')




if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)

# %%
