import  argparse
import json
from datetime import datetime
import os
import numpy as np

import torch

import models
import datasets
from utils import train_one_epoch, eval_on_dataloader

try:
    from tensorboardX import SummaryWriter
except:
    from torch.utils.tensorboard import SummaryWriter


def build_parser():

    parser = argparse.ArgumentParser(description="""This script generates Table 1 of the paper. 
    It is similar to train.py except for subtle differences such as that it checks for convergence rather than
    training a fixed number of epochs.""")
    parser.add_argument('title', type=str)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--convergence-epochs', type=int, default=3) # If the minimum val loss does not decrease in 3 epochs training will stop
    parser.add_argument('--convergence-accuracy-change-threshold', type=float, default=0.002)
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--checkpoint', default=None)
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
        device = torch.device(f'cuda:{args.gpu_id}')
        print("CUDA Recognized")
    else:
        device = torch.device('cpu')

    # Training with Random Initialization 
    overal_result = {}
    init_type = "random"
    dataset_result = {}
    for (dataset_name, num_classes) in [("cifar10", 10), ("cifar100", 100), ("svhn", 10)]:
        model_args = {
            "resnet18": {"num_classes": num_classes},
            "mlp": {"input_dim": 32 * 32 * 3, "num_classes": num_classes, 'activation':'tanh', 'bias':True},
            "logistic": {"input_dim": 32 * 32 * 3, "num_classes": num_classes},
        }
        
        optimizer_result = {}
        for optimizer_name in ["adam", "sgd", "sgd-momentum"]:
            model_result = {}
            for model_name in ["mlp", "logistic", "resnet18"]:
                print(f"Training model {model_name} on {dataset_name} with {optimizer_name} optimizer.")
                torch.manual_seed(args.random_seed)
                np.random.seed(args.random_seed)
                model = models.get_model(model_name, **model_args[model_name]).to(device)
                loaders = datasets.get_dataset(dataset_name)
                criterion = torch.nn.CrossEntropyLoss()

                if optimizer_name == "adam":
                    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
                elif optimizer_name == "sgd-momentum":
                    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
                else:
                    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
            
                train_accuracies = []
                stop_indicator = False
                epoch = 0
                while(not stop_indicator):
                    if epoch % 5 == 0:
                        print(f"\t Training in epoch {epoch + 1} \t")
                    train_loss, train_accuracy = train_one_epoch(device, model, optimizer, criterion, loaders['train_loader'])
                    train_loss, train_accuracy = eval_on_dataloader(device, criterion, model, loaders['train_loader'])

                    train_accuracies.append(train_accuracy)
                    epoch += 1
                    if train_accuracy >= 0.99:
                        print("Convergence codition met. Training accuracy > 0.99")
                        stop_indicator = True
                    
                    if len(train_accuracies) >= args.convergence_epochs:
                        if np.std(train_accuracies[-args.convergence_epochs:]) < args.convergence_accuracy_change_threshold:
                            print(f"\tConvergence codition met. Training accuracy = {train_accuracy} stopped improving")
                            stop_indicator = True

                test_loss, test_accuracy =  eval_on_dataloader(device, criterion, model, loaders['test_loader'])
                print(f"\tTest accuracy = {test_accuracy}")
                model_result[model_name] = test_accuracy
                
            optimizer_result[optimizer_name] = model_result
        dataset_result[dataset_name] = optimizer_result
    overal_result[init_type] = dataset_result

    
    # Training with warm-start
    init_type = "warm-start"
    dataset_result = {}
    for (dataset_name, num_classes) in [("cifar10", 10), ("cifar100", 100), ("svhn", 10)]:
        model_args = {
            "resnet18": {"num_classes": num_classes},
            "mlp": {"input_dim": 32 * 32 * 3, "num_classes": num_classes},
            "logistic": {"input_dim": 32 * 32 * 3, "num_classes": num_classes},
        }
        
        optimizer_result = {}
        for optimizer_name in ["adam", "sgd", "sgd-momentum"]:
            model_result = {}
            for model_name in ["mlp", "logistic", "resnet18"]:
                print(f"Training model {model_name} on half of {dataset_name} with {optimizer_name} optimizer.")
                torch.manual_seed(args.random_seed)
                np.random.seed(args.random_seed)
                model = models.get_model(model_name, **model_args[model_name]).to(device)
                loaders = datasets.get_dataset(f"half_{dataset_name}")
                criterion = torch.nn.CrossEntropyLoss()

                if optimizer_name == "adam":
                    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
                elif optimizer_name == "sgd-momentum":
                    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
                else:
                    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
                    
                train_accuracies = []
                stop_indicator = False
                epoch = 0
                while(not stop_indicator):
                    if epoch % 5 == 0:
                        print(f"\tPre-training in epoch {epoch + 1}")
                    train_loss, train_accuracy = train_one_epoch(device, model, optimizer, criterion, loaders['train_loader'])
                    train_loss, train_accuracy = eval_on_dataloader(device, criterion, model, loaders['train_loader'])
                    
                    train_accuracies.append(train_accuracy)
                    epoch += 1
                    if train_accuracy >= 0.99:
                        print("Convergence codition met. Training accuracy > 0.99")
                        stop_indicator = True
                    
                    if len(train_accuracies) >= args.convergence_epochs:
                        if np.std(train_accuracies[-args.convergence_epochs:]) < args.convergence_accuracy_change_threshold:
                            print(f"\tConvergence codition met. Training accuracy = {train_accuracy} stopped improving")
                            stop_indicator = True
                            
                loaders = datasets.get_dataset(f"{dataset_name}")
                criterion = torch.nn.CrossEntropyLoss()
                train_accuracies = []
                stop_indicator = False
                epoch = 0
                while(not stop_indicator):
                    if epoch % 5 == 0:
                        print(f"\t Training in epoch {epoch + 1}")
                    train_loss, train_accuracy = train_one_epoch(device, model, optimizer, criterion, loaders['train_loader'])
                    train_loss, train_accuracy = eval_on_dataloader(device, criterion, model, loaders['train_loader'])
                    
                    train_accuracies.append(train_accuracy)
                    epoch += 1
                    if train_accuracy >= 0.99:
                        print("Convergence codition met. Training accuracy > 0.99")
                        stop_indicator = True
                    
                    if len(train_accuracies) >= args.convergence_epochs:
                        if np.std(train_accuracies[-args.convergence_epochs:]) < args.convergence_accuracy_change_threshold:
                            print(f"\tConvergence codition met. Training accuracy = {train_accuracy} stopped improving")
                            stop_indicator = True

                test_loss, test_accuracy =  eval_on_dataloader(device, criterion, model, loaders['test_loader'])
                print(f"\tTest accuracy = {test_accuracy}")
                model_result[model_name] = test_accuracy
                
            optimizer_result[optimizer_name] = model_result
        dataset_result[dataset_name] = optimizer_result
    overal_result[init_type] = dataset_result

                           
    np.save(f"tables/table1-svhn-seed{args.random_seed}.npy", overal_result)
                

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)