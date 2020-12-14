import numpy as np
with open("tables/table1-template.txt", 'r') as f:
    table1_template = "".join(f.readlines())

temp_tables = [np.load(f"tables/table1-seed{seed}.npy", allow_pickle=True).item() for seed in [42, 43, 44, 45, 46]]


for i1, dataset in enumerate(['cifar10', 'svhn', 'cifar100']):
    for i2, init in enumerate(["random", "warm-start"]):
        for i3, model in enumerate(['resnet18', 'mlp', 'logistic']):
            for i4, opt in enumerate(['sgd', 'adam', 'sgd-momentum']):
                accs = [100 * temp_tables[i][init][dataset][opt][model] for i in range(len(temp_tables))]
                mean_acc = np.mean(accs)
                std_acc = np.std(accs)
                conf_int = std_acc / np.sqrt(len(temp_tables))
                table1_template = table1_template.replace(f"{i1 + 1}.{i2 + 1}.{i3 + 1}.{i4 + 1}", f"{mean_acc:2.1f} ({conf_int:0.1f})")
                
with open("tables/table1.tex", "w") as f:
    f.write(table1_template)