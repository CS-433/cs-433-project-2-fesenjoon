import  argparse
import json
from datetime import datetime
import os
import numpy as np

import torch

import models
import datasets
from utils import eval_on_dataloader, train_one_epoch

try:
    from tensorboardX import SummaryWriter
except:
    from torch.utils.tensorboard import SummaryWriter


def build_parser():

    parser = argparse.ArgumentParser(description="""This script generates Figure 2 of the original paper. 
    It is similar to a combination of train.py and train_online.py. 
    There are subtle differences, including: 
        1- train.py does not have a convergence threshold.
        2- Convergence threshold is checked slightly differently than train_online.py""")
    parser.add_argument('title', type=str)
    parser.add_argument('--model', type=str, default='resnet18', choices=models.get_available_models())
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'svhn'])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--acc-threshold', type=float, default=0.99)

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

    try:
        summary_writer = SummaryWriter(logdir=experiment_dir)
    except:
        summary_writer = SummaryWriter(experiment_dir)
    
    print("Starting Online Learning")
    #Online learning setup
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    model = models.get_model(args.model).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    loaders = datasets.get_dataset(f"online_with_val_{args.dataset}", split_size=args.split_size)
    number_of_samples_online = []
    test_accuracies_online = []
    training_times_online = []
    for i, train_loader in enumerate(loaders['train_loaders']):
        t_start = datetime.now()
        n_train = (i + 1) * args.split_size
        number_of_samples_online.append(n_train)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        print(f"Warm-Start Training with {n_train} data.")
        
        train_accuracies = []
        stop_indicator = False
        epoch = 0
        while(not stop_indicator):
            if epoch % 5 == 0:
                print(f"Starting training in epoch {epoch + 1}")
            train_loss, train_accuracy = train_one_epoch(device, model, optimizer, criterion,
                                                         train_loader)
            train_loss, train_accuracy =  eval_on_dataloader(device, criterion, model, train_loader)
            train_accuracies.append(train_accuracy)
            epoch += 1
            
            if train_accuracy >= args.acc_threshold:
                print(f"Convergence codition met. Training accuracy > {100 * args.acc_threshold}")
                stop_indicator = True


                    
        t_end = datetime.now()
        training_time = (t_end - t_start).total_seconds()
        test_loss, test_accuracy =  eval_on_dataloader(device, criterion, model, loaders['test_loader'])
        test_accuracies_online.append(test_accuracy)
        training_times_online.append(training_time)
        summary_writer.add_scalar("test_accuracy_online", test_accuracy, n_train)
        summary_writer.add_scalar("train_time_online", training_time, n_train)
        
        

    
    
    print("Starting Offline Learning")
    # Offline learning setup
    n_experiments = len(loaders['train_loaders'])
    number_of_samples_offline = []
    test_accuracies_offline = []
    training_times_offline = []
    for i in range(1, n_experiments + 1):
        t_start = datetime.now()
        n_train = i * args.split_size
        number_of_samples_offline.append(n_train)
        print(f"Running {i}_th experiment with Train size = {n_train}")
        
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
        while(not stop_indicator):
            if epoch % 5 == 0:
                print(f"Starting training in epoch {epoch + 1}")
            train_loss, train_accuracy = train_one_epoch(device, model, optimizer, criterion,
                                                         loaders['train_loader'])
#             val_loss, val_accuracy =  eval_on_dataloader(model, loaders['val_loader'])
            train_loss, train_accuracy = eval_on_dataloader(device, criterion, model,
                                                            loaders['train_loader'])  # To get model's final accuracy
            train_accuracies.append(train_accuracy)
            epoch += 1
            
            if train_accuracy >= args.acc_threshold:
                print(f"Convergence codition met. Training accuracy > {100 * args.acc_threshold}")
                stop_indicator = True

            
                    
        t_end = datetime.now()
        training_time = (t_end - t_start).total_seconds()
        test_loss, test_accuracy =  eval_on_dataloader(device, criterion, model, loaders['test_loader'])
        test_accuracies_offline.append(test_accuracy)
        training_times_offline.append(training_time)
        summary_writer.add_scalar("test_accuracy_offline", test_accuracy, n_train)
        summary_writer.add_scalar("train_time_offline", training_time, n_train)
        

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    number_of_samples_online = np.array(number_of_samples_online) / 1000
    number_of_samples_offline = np.array(number_of_samples_offline) / 1000
    axs[0].plot(number_of_samples_online, test_accuracies_online, label='warm start', color='C0')
    axs[0].plot(number_of_samples_offline, test_accuracies_offline, label='random', color='C1')
    axs[0].set_ylabel("Tets Accuracy")
    axs[0].set_xlabel("Number of Samples (thousands)")
    axs[1].plot(number_of_samples_online, training_times_online, label='warm start', color='C0')
    axs[1].plot(number_of_samples_offline, training_times_offline, label='random', color='C1')
    axs[1].set_ylabel("Train Time (seconds)")
    axs[1].set_xlabel("Number of Samples (thousands)")
    plt.legend()
    plt.savefig(f"figures/figure2-{args.dataset}-{100 * args.acc_threshold}.pdf")
#     np.save(f"tables/figure2-{args.dataset}-{100 * args.acc_threshold}.npy", {
#         "test_acc_online": test_accuracies_online,
#         "test_acc_offline": test_accuracies_offline,
#         "train_time_online": training_times_online,
#         "train_time_offline": training_times_offline
#     })
            
            
            
    

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)