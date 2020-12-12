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
    parser.add_argument('--exp-dir', default=None)
    return parser


def main(args):

    with open(os.path.join(args.exp_dir, 'config.json')) as f:
        args.__dict__.update(json.load(f))


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
    loaders = datasets.get_dataset(args.dataset, **dataset_args)
    num_classes = loaders.get('num_classes', 10)
    model_args = {}
    if args.model == 'mlp':
        model_args['bias'] = args.mlp_bias
        model_args['activation'] = args.mlp_activation
        model_args["input_dim"] =  32 * 32 * 3
    model = models.get_model(args.model, num_classes=num_classes, **model_args).to(device)
    model.load_state_dict(torch.load(os.path.join(args.exp_dir, 'final.pt'), map_location=device)['model'])
    criterion = torch.nn.CrossEntropyLoss()
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


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)