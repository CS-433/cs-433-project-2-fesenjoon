import  argparse
import json
from datetime import datetime
import os
import numpy as np
import time
import torch

import models
import datasets
from utils import eval_on_dataloader, train_one_epoch


def train(lr=0.1, batch_size=64, max_epoch=700, rs=7, save=False, title='0', outdir='fig3', resume_model=None, resume_epoch=0, half_dataset=False):

    #experiment_dir = outdir
    experiment_dir = os.path.join('exp', title, datetime.now().strftime('%b%d_%H-%M-%S'))
    os.makedirs(experiment_dir, exist_ok=True)
    # Set the seed
    torch.manual_seed(rs)
    np.random.seed(rs)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        #print("CUDA Recognized")
    else:
        device = torch.device('cpu')

    model = models.get_model('resnet18').to(device)

    if resume_model is not None:
        model = resume_model
     
    loaders = datasets.get_dataset('first_half_cifar10' if half_dataset else 'cifar10', batch_size=batch_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
   
    start = time.time()

    for epoch in range(resume_epoch, max_epoch + 1):
        print(f"Epoch {epoch}")
        train_loss, train_accuracy = train_one_epoch(device, model, optimizer, criterion, loaders["train_loader"])
        print("Train accuracy: {} Train loss: {}".format(train_accuracy, train_loss))
        test_accuracy = 0
        
        if not half_dataset:
            test_loss, test_accuracy = eval_on_dataloader(device, criterion, model, loaders['test_loader'])
            print("Test accuracy: {} Test loss: {}".format(test_accuracy, test_loss))

        if train_accuracy > 0.99:
            cost = time.time() - start
            return train_accuracy, test_accuracy, cost, model

    
    return 0, 0, 0, None
    #torch.save({
    #    'model': model.state_dict()
    #}, os.path.join(experiment_dir, 'final.pt'))

def build_parser():

    parser = argparse.ArgumentParser(description="""This scripts generates Figure 3 of the original paper.
    It performs training in the offline setting with various learning rate and batch sizes.""")
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--index_start', type=int, default=0)
    parser.add_argument('--index_stop', type=int, default=100)
    return parser


def main(args):
    bsizes = [16, 32, 64, 128]
    lrs = [0.001, 0.01, 0.1]
    os.makedirs('exp/warm/', exist_ok=True)
    os.makedirs('exp/random/', exist_ok=True)

    for i in range(args.index_start, args.index_stop + 1):
        np.random.seed(i + args.random_seed)
        for bsize in bsizes:
            for lr in lrs:
                rs = np.random.randint(0, 100000)
                train_accuracy, test_accuracy, cost, init_model = train(lr=lr, batch_size=bsize, rs=rs, half_dataset=True)
                for j in range(3):
                    rs = np.random.randint(0, 100000)
                    lr_rand = np.random.randint(0, 3)
                    bsize_rand = np.random.randint(0, 4)
                    train_accuracy, test_accuracy, cost, model = train(lr=lrs[lr_rand], batch_size=bsizes[bsize_rand], rs=rs, resume_model=init_model)
                    np.savetxt('exp/warm/' + str(i) + '_' + str(lr) + '_' + str(bsize) + '_' + str(j) + '.csv', (train_accuracy, test_accuracy, cost))
                    
                    rs = np.random.randint(0, 100000)
                    print(i, j, lr, bsize, rs)
                    train_accuracy, test_accuracy, cost, model = train(lr=lr, batch_size=bsize, rs=rs)
                    np.savetxt('exp/random/' + str(i) + '_' + str(lr) + '_' + str(bsize) + '_' + str(j) + '.csv', (train_accuracy, test_accuracy, cost))




if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)