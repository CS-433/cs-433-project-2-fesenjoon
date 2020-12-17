import  argparse
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
    parser.add_argument('--title', type=str)
    parser.add_argument('--exp-dir', type=str, default=None)
    parser.add_argument('--model', type=str, default='resnet18', choices=models.get_available_models())
#     parser.add_argument('--dataset', type=str, default='cifar10', choices=datasets.get_available_datasets())
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--split-size', type=int, default=5000)
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--convergence-epochs', type=int, default=5) # If the minimum val loss does not decrease in 3 epochs training will stop
#     parser.add_argument('--save-per-epoch', action='store_true', default=False)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--checkpoint-shrink', default=1.0, type=float)
    parser.add_argument('--checkpoint-perturb', default=0.0, type=float)
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
    if args.exp_dir:
        experiment_dir = args.exp_dir
    else:
        experiment_dir = os.path.join('exp', args.title, experiment_time)
    os.makedirs(experiment_dir, exist_ok=True)
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
    
    print("Starting Online Learning")
    #Online learning setup
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    model = models.get_model(args.model).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    loaders = datasets.get_dataset("online_with_val_cifar10", split_size=args.split_size)
    number_of_samples_online = []
    test_accuracies_online = []
    training_times_online = []
    epoch = 0
    for i, train_loader in enumerate(loaders['train_loaders']):
        t_start = datetime.now()
        n_train = (i + 1) * args.split_size
        number_of_samples_online.append(n_train)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        random_model = models.get_model(args.model).to(device)
        with torch.no_grad():
            for real_parameter, random_parameter in zip(model.parameters(), random_model.parameters()):
                real_parameter.mul_(args.checkpoint_shrink).add_(random_parameter, alpha=args.checkpoint_perturb)
        
        train_accuracies = []
        while True:
            if epoch % 5 == 0:
                print(f"Starting training in epoch {epoch + 1}")
            train_loss, train_accuracy = train_one_epoch(model, optimizer, criterion, train_loader)
            val_loss, val_accuracy =  eval_on_dataloader(model, loaders['val_loader'])
            test_loss, test_accuracy = eval_on_dataloader(model, loaders['test_loader'])
            train_accuracies.append(train_accuracy)
            epoch += 1
            summary_writer.add_scalar("test_accuracy", test_accuracy, epoch)
            summary_writer.add_scalar("test_loss", test_loss, epoch)
            summary_writer.add_scalar("train_accuracy", train_accuracy, epoch)
            summary_writer.add_scalar("train_loss", train_loss, epoch)
            summary_writer.add_scalar("val_accuracy", val_accuracy, epoch)
            summary_writer.add_scalar("val_loss", val_loss, epoch)
            #if len(train_accuracies) >= args.convergence_epochs and \
            #        max(train_accuracies) not in train_accuracies[-args.convergence_epochs:]:
            if train_accuracy >= 0.99:
                print("Convergence condition met")
                break

        val_loss, val_accuracy = eval_on_dataloader(model, loaders['val_loader'])
        test_loss, test_accuracy = eval_on_dataloader(model, loaders['test_loader'])
        summary_writer.add_scalar("online_val_accuracy", val_accuracy, n_train)
        summary_writer.add_scalar("online_val_loss", val_loss, n_train)
        summary_writer.add_scalar("online_test_accuracy", test_accuracy, n_train)
        summary_writer.add_scalar("online_test_loss", test_loss, n_train)
        t_end = datetime.now()
        training_time = (t_end - t_start).total_seconds()

        training_times_online.append(training_time)
        summary_writer.add_scalar("online_train_time", training_time, n_train)

    summary_writer.close()

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)