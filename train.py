import  argparse
import json
from datetime import datetime
import os
import numpy as np

import torch

import models
import datasets

parser = argparse.ArgumentParser()
parser.add_argument('title', type=str)
parser.add_argument('--model', type=str, default='resnet18', choices=models.get_available_models())
parser.add_argument('--dataset', type=str, default='cifar10', choices=datasets.get_available_datasets())
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--random-seed', type=int, default=42)


args = parser.parse_args()

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
else:
    device = torch.device('cpu')


def get_accuracy(logit, true_y):
    pred_y = torch.argmax(logit, dim=1)
    return (pred_y == true_y).sum() / len(true_y)


model = models.get_model(args.model).to(device)
loaders = datasets.get_dataset(args.dataset)
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
criterion = torch.nn.CrossEntropyLoss()

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
        optimizer.step()

        accuracies.append(batch_accuracy)
        losses.append(loss)

    print("Train accuracy: {} Train loss: {}".format(np.mean(accuracies), np.mean(losses)))

    accuracies = []
    losses = []
    for batch_idx, (data_x, data_y) in enumerate(loaders["test_loader"]):
        data_x = data_x.to(device)
        data_y = data_y.to(device)

        model_y = model(data_x)
        loss = criterion(model_y, data_y)
        batch_accuracy = get_accuracy(model_y, data_y)

        accuracies.append(batch_accuracy)
        losses.append(loss)

    print("Test accuracy: {} Test loss: {}".format(np.mean(accuracies), np.mean(losses)))

    torch.save({
        'model': model.state_dict()
    }, os.path.join(experiment_dir, f'chkpt_epoch{epoch}.pt'))