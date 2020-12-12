import  argparse
import json
from datetime import datetime
import os
import numpy as np

import torch

import models
import datasets
from tensorboardX import SummaryWriter

def build_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('title', type=str)
    parser.add_argument('--model', type=str, default='resnet18', choices=models.get_available_models())
    parser.add_argument('--dataset', type=str, default='cifar10', choices=datasets.get_available_datasets())
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lr-schedule', action='store_true', default=False)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--save-per-epoch', action='store_true', default=False)
    parser.add_argument('--grad-track', action='store_true', default=False)
    parser.add_argument('--data-aug', action='store_true', default=False)
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

    experiment_dir = os.path.join('exp', args.title, datetime.now().strftime('%b%d_%H-%M-%S'))
    os.makedirs(experiment_dir)
    with open(os.path.join(experiment_dir, "config.json"), "w") as f:
        json.dump(args_dict, f, indent=4, sort_keys=True, default=lambda x: x.__name__)

    # Set the seed
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print("CUDA Recognized")
    else:
        device = torch.device('cpu')

    def get_accuracy(logit, true_y):
        pred_y = torch.argmax(logit, dim=1)
        return (pred_y == true_y).sum() / len(true_y)

    model = models.get_model(args.model).to(device)
    loaders = datasets.get_dataset(args.dataset, data_aug=args.data_aug)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    if args.lr_schedule:
        def lambda1(epoch):
            if epoch / args.epochs < 0.33:
                return args.lr
            elif epoch / args.epochs < 0.66:
                return args.lr * 0.1
            else:
                return args.lr * 0.01
            
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    criterion = torch.nn.CrossEntropyLoss()
    summary_writer = SummaryWriter(logdir=experiment_dir)

    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device)['model'])
    
    dummy_model = models.get_model(args.model).to(device)
    with torch.no_grad():
        for real_parameter, random_parameter in zip(model.parameters(), dummy_model.parameters()):
            real_parameter.mul_(args.checkpoint_shrink).add_(random_parameter, alpha=args.checkpoint_perturb)

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}")
        accuracies = []
        losses = []
        for batch_idx, (data_x, data_y) in enumerate(loaders["train_loader"]):
            data_x = data_x.to(device)
            data_y = data_y.to(device)

            optimizer.zero_grad()
            model_y = model(data_x)
            loss = criterion(model_y, data_y)
            batch_accuracy = get_accuracy(model_y, data_y)
            loss.backward()
            
            if (args.grad_track):
                grads = []
                for param in model.parameters():
                    grads.append(param.grad.view(-1))
                grads = torch.cat(grads)
                summary_writer.add_scalar("grad_norm", torch.norm(grads), batch_idx + (epoch - 1) * len(loaders["train_loader"]))
            
            optimizer.step()
            accuracies.append(batch_accuracy.item())
            losses.append(loss.item())
            if args.lr_schedule:
                scheduler.step()

        train_loss = np.mean(losses)
        train_accuracy = np.mean(accuracies)
        print("Train accuracy: {} Train loss: {}".format(train_accuracy, train_loss))
        summary_writer.add_scalar("train_loss", train_loss, epoch)
        summary_writer.add_scalar("train_accuracy", train_accuracy, epoch)

        accuracies = []
        losses = []
        for batch_idx, (data_x, data_y) in enumerate(loaders["test_loader"]):
            data_x = data_x.to(device)
            data_y = data_y.to(device)

            model_y = model(data_x)
            loss = criterion(model_y, data_y)
            batch_accuracy = get_accuracy(model_y, data_y)

            accuracies.append(batch_accuracy.item())
            losses.append(loss.item())

        test_loss = np.mean(losses)
        test_accuracy = np.mean(accuracies)
        print("Test accuracy: {} Test loss: {}".format(test_accuracy, test_loss))
        summary_writer.add_scalar("test_loss", test_loss, epoch)
        summary_writer.add_scalar("test_accuracy", test_accuracy, epoch)

        if args.save_per_epoch:
            torch.save({
                'model': model.state_dict()
            }, os.path.join(experiment_dir, f'chkpt_epoch{epoch}.pt'))

    torch.save({
        'model': model.state_dict()
    }, os.path.join(experiment_dir, 'final.pt'))


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)