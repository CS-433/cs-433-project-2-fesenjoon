import os

import numpy
import numpy as np
import tensorflow as tf
import torch

root_dir = os.path.dirname(__file__)


def get_data_for_runs(runs, exp_dir=os.path.join(root_dir, "exp")):
    summaries = {}

    for run_dir, title in runs.items():
        if not os.path.exists(run_dir):
            run_dir = os.path.join(exp_dir, run_dir)
        for file in os.listdir(run_dir):
            if "events.out" in file:
                #print(file)
                if title not in summaries:
                    summaries[title] = []
                summaries[title].append(os.path.join(run_dir, file))
                break

    datas = {}
    for i, title in enumerate(summaries):
        for summary_path in summaries[title]:
            try:
                for summary in tf.compat.v1.train.summary_iterator(summary_path):
                    for v in summary.summary.value:
                        if v.tag not in datas:
                            datas[v.tag] = [list() for _ in range(len(summaries))]
                        datas[v.tag][i].append((summary.wall_time, v.simple_value))
            except Exception as e:
                print(e)
    for tag in datas:
        for i in range(len(datas[tag])):
            datas[tag][i] = np.array(list(zip(*sorted(datas[tag][i])))[1])
    return datas, summaries


def get_accuracy(logit, true_y):
    pred_y = torch.argmax(logit, dim=1)
    return (pred_y == true_y).float().mean()
    return torch.true_divide((pred_y == true_y).sum(), len(true_y))

def eval_on_dataloader(device, criterion, model, dataloader):
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


def train_one_epoch(device, model, optimizer, criterion, train_dataloader):
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

