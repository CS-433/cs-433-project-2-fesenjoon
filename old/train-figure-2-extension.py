import argparse
import json
from datetime import datetime
import os
import numpy as np

import torch

import models
import datasets

try:
    from tensorboardX import SummaryWriter
except:
    from torch.utils.tensorboard import SummaryWriter


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('title', type=str)
    parser.add_argument('--model', type=str, default='resnet18', choices=models.get_available_models())
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'svhn'])
    parser.add_argument('--lr', type=float, default=0.001)

    parser.add_argument('--split-size', type=int, default=5000)
    parser.add_argument('--random-seed', type=int, default=42)
    return parser


def main(args):
    print("Running with arguments:")
    args_dict = {}
    for key in vars(args):
        if key == "default_function":
            continue
        args_dict[key] = getattr(args, key)
        print(key, ": ", args_dict[key])
    print("---")

    experiment_time = datetime.now().strftime('%b%d_%H-%M-%S')
    experiment_dir = os.path.join('exp', args.title, experiment_time)
    os.makedirs(experiment_dir)
    with open(os.path.join(experiment_dir, "config.json"), "w") as f:
        json.dump(args_dict, f, indent=4, sort_keys=True, default=lambda x: x.__name__)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print("CUDA Recognized")
    else:
        device = torch.device('cpu')

    def get_accuracy(logit, true_y):
        pred_y = torch.argmax(logit, dim=1)
        return torch.true_divide((pred_y == true_y).sum(), len(true_y))

    def train_one_epoch(model, optimizer, criterion, train_dataloader):
        accuracies = []
        losses = []
        for batch_idx, (data_x, data_y) in enumerate(train_dataloader):
            data_x = data_x.to(device)
            data_y = data_y.to(device)

            optimizer.zero_grad()
            model_y = model(data_x)
            loss = criterion(model_y, data_y)
            batch_accuracy = get_accuracy(model_y, data_y)
            loss.backward()
            optimizer.step()

            accuracies.append(batch_accuracy.item())
            losses.append(loss.item())

        train_loss = np.mean(losses)
        train_accuracy = np.mean(accuracies)
        return train_loss, train_accuracy

    def eval_on_dataloader(model, dataloader):
        accuracies = []
        losses = []
        for batch_idx, (data_x, data_y) in enumerate(dataloader):
            data_x = data_x.to(device)
            data_y = data_y.to(device)

            model_y = model(data_x)
            loss = criterion(model_y, data_y)
            batch_accuracy = get_accuracy(model_y, data_y)

            accuracies.append(batch_accuracy.item())
            losses.append(loss.item())

        loss = np.mean(losses)
        accuracy = np.mean(accuracies)
        return loss, accuracy

    try:
        summary_writer = SummaryWriter(logdir=experiment_dir)
    except:
        summary_writer = SummaryWriter(experiment_dir)

#     acc_thresholds = np.linspace(0.81, 0.99, 10)
    acc_thresholds = np.linspace(0.71, 0.79, 5)

    test_accuracies_online = []
    test_accuracies_offline = []
    for acc_threshold in acc_thresholds:
        print("Starting Online Learning")
        # Online learning setup
        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)
        model = models.get_model(args.model).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        loaders = datasets.get_dataset(f"online_with_val_{args.dataset}", split_size=args.split_size)
        for i, train_loader in enumerate(loaders['train_loaders']):
            n_train = (i + 1) * args.split_size
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            if i % 5 == 0:
                print(f"Warm-Start Training with {n_train} data.")

            train_accuracies = []
            stop_indicator = False
            epoch = 0
            while not stop_indicator:
                train_loss, train_accuracy = train_one_epoch(model, optimizer, criterion, train_loader)
                train_loss, train_accuracy = eval_on_dataloader(model, train_loader)
                train_accuracies.append(train_accuracy)
                epoch += 1

                if train_accuracy >= acc_threshold:
                    stop_indicator = True

        test_loss, test_accuracy = eval_on_dataloader(model, loaders['test_loader'])
        test_accuracies_online.append(test_accuracy)
        summary_writer.add_scalar("warm start accuracy", test_accuracy, acc_threshold)

        print("Starting Offline Learning")
        # Offline learning setup
        n_experiments = len(loaders['train_loaders'])
        n_train = n_experiments * args.split_size

        # Set the seed
        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)
        loaders = datasets.get_dataset(f"partial_with_val_{args.dataset}", n_train)
        model = models.get_model(args.model).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = torch.nn.CrossEntropyLoss()

        train_accuracies = []
        stop_indicator = False
        epoch = 0
        while not stop_indicator:
            train_loss, train_accuracy = train_one_epoch(model, optimizer, criterion, loaders['train_loader'])
            #             val_loss, val_accuracy =  eval_on_dataloader(model, loaders['val_loader'])
            train_loss, train_accuracy = eval_on_dataloader(model,
                                                            loaders['train_loader'])  # To get model's final accuracy
            train_accuracies.append(train_accuracy)
            epoch += 1

            if train_accuracy >= acc_threshold:
                stop_indicator = True

        test_loss, test_accuracy = eval_on_dataloader(model, loaders['test_loader'])
        test_accuracies_offline.append(test_accuracy)
        summary_writer.add_scalar("random init accuracy", test_accuracy, acc_threshold)

    import matplotlib.pyplot as plt
    plt.plot(acc_thresholds, test_accuracies_online, label='warm start', color='C0')
    plt.plot(acc_thresholds, test_accuracies_offline, label='random', color='C1')
    plt.ylabel("Test Accuracy")
    plt.xlabel("Convergence accuracy threshold")
    plt.legend()
    plt.savefig(f"figures/figure2-{args.dataset}-convergence.pdf")
#     np.save(f"tables/figure2-{args.dataset}-extended.npy", {
#         "test_acc_online": test_accuracies_online,
#         "test_acc_offline": test_accuracies_offline,
#     })


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)