#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
from utils import get_data_for_runs
import glob


ws_path = glob.glob('exp/grad_track_ws/*')[0]
ws_path = '/'.join(ws_path.split('/')[1:])

aug_path = glob.glob('exp/grad_track_aug/*')[0]
aug_path = '/'.join(aug_path.split('/')[1:])

runs = {ws_path: "Warm start", aug_path: "Aug start"}

datas, summaries = get_data_for_runs(runs)

ws_first = datas['grad_norm_first'][0]
ws_second = datas['grad_norm_second'][0]

aug_first = datas['grad_norm_first'][1]
aug_second = datas['grad_norm_second'][1]

plt.figure(figsize= (5,5))

plt.plot(range(0,30), ws_first, label='WS old data')
plt.plot(range(0,30), ws_second, label='WS new data')
plt.plot(range(0,30), aug_first, label='DA old data')
plt.plot(range(0,30), aug_second, label='DA new data')
plt.legend()
    
plt.savefig("fig5_aug.png", dpi=600)


