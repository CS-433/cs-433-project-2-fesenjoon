import matplotlib.pyplot as plt
import numpy as np
from utils import get_data_for_runs


datas, _ = get_data_for_runs({"exp/half_cifar/Dec12_21-44-54": "Half"})

plt.subplots(1, 2, figsize=(6, 3))
plt.subplot(1, 2, 1)
tag = 'test_accuracy'
x = np.arange(0, len(datas[tag][0]))
y = datas[tag][0]
y = y * 100
plt.plot(x, y)
plt.ylabel('Test Accuracy')
plt.xlabel('Epoch')

runs = {
    "exp/cifar/": "0",
    "exp/cifar10_warmup_20epoch/": "20",
    "exp/cifar10_warmup_40epoch/": "40",
    "exp/cifar10_warmup_60epoch/": "60",
    "exp/cifar10_warmup_80epoch/": "80",
    "exp/cifar10_warmup_100epoch/": "100",
    "exp/cifar10_warmup_120epoch/": "120",
    "exp/cifar10_warmup_140epoch/": "140",
    "exp/cifar10_warmup_160epoch/": "160",
    "exp/cifar10_warmup_180epoch/": "180",
    "exp/cifar10_warmup_200epoch/": "200"
}


plt.subplot(1, 2, 2)
datas, summaries = get_data_for_runs(runs)
points = []
for i, title in enumerate(summaries):
    if title == '0':
        normal_accuracy = datas['test_accuracy'][i][-1]
        break
print(summaries)
for i, title in enumerate(summaries):
    print(title, datas['test_accuracy'][i][-1])
    points.append((int(title), (datas['test_accuracy'][i][-1] - normal_accuracy) / normal_accuracy * 100))

x, y = zip(*sorted(points))
plt.plot(x, y)
plt.ylabel('Percent Difference')
plt.xlabel('Start Epoch')
plt.tight_layout()
plt.savefig('figure4.pdf')