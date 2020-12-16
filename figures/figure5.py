#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
from utils import get_data_for_runs
import glob


ws_path = glob.glob('exp/grad_track_ws/*')[0]
ws_path = '/'.join(ws_path.split('/')[1:])

runs = {ws_path: "Warm start"}

for sp in ['0.3', '0.5', '0.7', '0.9']:
    mask = 'exp/grad_track_sp_' + sp + '/*'
    path = glob.glob(mask)
    path = '/'.join(path[0].split('/')[1:])
    runs[path] = 'Shrink perturb lambda = ' + sp

datas, summaries = get_data_for_runs(runs)

ws_first = datas['grad_norm_first'][0]
ws_second = datas['grad_norm_second'][0]

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for i, sp in enumerate(['0.3', '0.5', '0.7', '0.9']):
    ax = axes[i]
    ax.plot(range(0,30), ws_first, label='WS old data')
    ax.plot(range(0,30), ws_second, label='WS new data')
    sp_first = datas['grad_norm_first'][i+1]
    sp_second = datas['grad_norm_second'][i+1]
    ax.plot(range(0,30), sp_first, label='SP old data, $\lambda$ = ' + sp)
    ax.plot(range(0,30), sp_second, label='SP new data, $\lambda$ = ' + sp)
    ax.set(xlabel='Training Epoch', ylabel='Average Gradient')
    ax.legend()
    
plt.savefig("fig5_shrink_pertrub.png", dpi=600)


