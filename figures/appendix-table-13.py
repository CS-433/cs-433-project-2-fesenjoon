from utils import get_data_for_runs

for weight_decay in [0.1, 0.01, 0.001]:
    runs = {
        "exp/half_cifar10_then_cifar10_{}_weight_decay/".format(weight_decay): "Warm Starting",
        "exp/cifar10_{}_weight_decay/".format(weight_decay): "Random Init",
    }

    datas, summaries = get_data_for_runs(runs)
    for i, title in enumerate(summaries):
        print(weight_decay, title, datas['test_accuracy'][i][-1])