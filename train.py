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
    parser.add_argument('title', type=str)
    parser.add_argument('--exp-dir', type=str, default=None)
    parser.add_argument('--model', type=str, default='resnet18', choices=models.get_available_models())
    parser.add_argument('--dataset', type=str, default='cifar10', choices=datasets.get_available_datasets())
    parser.add_argument('--dataset-portion', type=float, required=False, default=None)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lr-schedule', action='store_true', default=False)
    parser.add_argument('--mlp-bias', action='store_true', default=False)
    parser.add_argument('--mlp-activation', type=str, default="relu")
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--grad-track', action='store_true', default=False)
    parser.add_argument('--data-aug', action='store_true', default=False)
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['adam', 'sgd'])
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--save-per-epoch', default=None, type=int)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--checkpoint-shrink', default=1.0, type=float)
    parser.add_argument('--checkpoint-perturb', default=0.0, type=float)
    parser.add_argument('--checkpoint-num-classes', default=None, type=int)
    return parser


def main(args, experiment_dir=None):
    print("Running with arguments:")
    args_dict = {}
    for key in vars(args):
        if key == "default_function":
            continue
        args_dict[key] = getattr(args, key)
        print(key, ": ", args_dict[key])
    print("---")

    if experiment_dir is None:
        if args.exp_dir is not None:
            experiment_dir = args.exp_dir
        else:
            experiment_dir = os.path.join('exp', args.title, datetime.now().strftime('%b%d_%H-%M-%S'))
    os.makedirs(experiment_dir, exist_ok=True)
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
        return torch.true_divide((pred_y == true_y).sum(), len(true_y))

    dataset_args = {}
    if args.dataset_portion is not None:
        dataset_args["dataset_portion"] = args.dataset_portion

    if args.grad_track:
        two_half_sets = datasets.get_dataset("two_half_cifar10", data_aug=args.data_aug, **dataset_args)
        first_loader = two_half_sets['train_loader_first']
        second_loader = two_half_sets['train_loader_second']

    #loaders = datasets.get_dataset(args.dataset)

    loaders = datasets.get_dataset(args.dataset, data_aug=args.data_aug, **dataset_args)
    num_classes = loaders.get('num_classes', 10)
    model_args = {}

    if args.model == 'mlp':
        model_args['bias'] = args.mlp_bias
        model_args['activation'] = args.mlp_activation
        model_args["input_dim"] =  32 * 32 * 3

    model = models.get_model(args.model, num_classes=num_classes, **model_args).to(device)

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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
    try:
        summary_writer = SummaryWriter(logdir=experiment_dir)
    except:
        summary_writer = SummaryWriter(experiment_dir)

    if args.checkpoint:
        real_fc = None
        if args.checkpoint_num_classes is not None and args.checkpoint_num_classes != num_classes:
            real_fc = model.fc
            model.fc = torch.nn.Linear(model.fc.in_features, args.checkpoint_num_classes)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device)['model'])
    
        if real_fc is not None:
            model.fc = real_fc
        dummy_model = models.get_model(args.model, num_classes=num_classes).to(device)
        with torch.no_grad():
            for real_parameter, random_parameter in zip(model.parameters(), dummy_model.parameters()):
                real_parameter.mul_(args.checkpoint_shrink).add_(random_parameter, alpha=args.checkpoint_perturb)

    for epoch in range(1, args.epochs + 1):
        model.train()
        print(f"Epoch {epoch}")
        accuracies = []
        losses = []
        if (args.grad_track):
            avg_grad = []
            for batch_idx, (data_x, data_y) in enumerate(first_loader):
                data_x = data_x.to(device)
                data_y = data_y.to(device)
                optimizer.zero_grad()
                model_y = model(data_x)
                loss = criterion(model_y, data_y)
                loss.backward()
                grads = []
                for param in model.parameters():
                    grads.append(param.grad.view(-1))
                grads = torch.cat(grads)
                grad_norm = torch.norm(grads)
                avg_grad.append(grad_norm)
            first_grad = torch.stack(avg_grad).mean()
            print('Grad for the first part: ',first_grad)
            summary_writer.add_scalar("grad_norm_first", first_grad, epoch)

            avg_grad = []
            for batch_idx, (data_x, data_y) in enumerate(second_loader):
                data_x = data_x.to(device)
                data_y = data_y.to(device)
                optimizer.zero_grad()
                model_y = model(data_x)
                loss = criterion(model_y, data_y)
                loss.backward()
                grads = []
                for param in model.parameters():
                    grads.append(param.grad.view(-1))
                grads = torch.cat(grads)
                grad_norm = torch.norm(grads)
                avg_grad.append(grad_norm)
            second_grad = torch.stack(avg_grad).mean()
            print('Grad for the second part: ',second_grad)
            summary_writer.add_scalar("grad_norm_second", second_grad, epoch)

        for batch_idx, (data_x, data_y) in enumerate(loaders["train_loader"]):
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
        print("Train accuracy: {} Train loss: {}".format(train_accuracy, train_loss))
        summary_writer.add_scalar("train_loss", train_loss, epoch)
        summary_writer.add_scalar("train_accuracy", train_accuracy, epoch)

        accuracies = []
        losses = []
        model.eval()
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
        
        if args.lr_schedule:
            scheduler.step()

        if args.save_per_epoch:
            if epoch % args.save_per_epoch == 0:
                torch.save({
                    'model': model.state_dict()
                }, os.path.join(experiment_dir, f'chkpt_epoch{epoch}.pt'))

    torch.save({
        'model': model.state_dict()
    }, os.path.join(experiment_dir, 'final.pt'))

    summary_writer.close()

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
