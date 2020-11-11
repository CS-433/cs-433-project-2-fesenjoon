import os

import torch
from torch.utils.data import random_split
from torchvision import transforms, datasets
import  numpy as np


def get_cifar10_loaders():
    normalize_transform = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_transform = transforms.Compose([transforms.ToTensor(), normalize_transform])
    test_transform = transforms.Compose([transforms.ToTensor(), normalize_transform])

    original_train_dataset = datasets.CIFAR10(root=os.path.join('data', 'cifar10_data'),
                                              train=True, transform=train_transform, download=True)
    original_test_dataset = datasets.CIFAR10(root=os.path.join('data', 'cifar10_data'),
                                             train=False, transform=test_transform, download=True)

    loader_args = {
        "batch_size": 256,

    }

    full_train_loader = torch.utils.data.DataLoader(
        dataset=original_train_dataset,
        shuffle=True,
        **loader_args)

    test_loader = torch.utils.data.DataLoader(
        dataset=original_test_dataset,
        shuffle=False,
        **loader_args)

def get_cifar10_half_loaders():
    normalize_transform = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_transform = transforms.Compose([transforms.ToTensor(), normalize_transform])
    test_transform = transforms.Compose([transforms.ToTensor(), normalize_transform])

    original_train_dataset = datasets.CIFAR10(root=os.path.join('data', 'cifar10_data'),
                                              train=True, transform=train_transform, download=True)
    original_test_dataset = datasets.CIFAR10(root=os.path.join('data', 'cifar10_data'),
                                             train=False, transform=test_transform, download=True)

    dataset_size = len(original_train_dataset)
    split = int(np.floor(0.5 * dataset_size))
    half_train_dataset, _ = random_split(original_train_dataset, [dataset_size - split, split])

    loader_args = {
        "batch_size": 256,

    }

    train_loader = torch.utils.data.DataLoader(
        dataset=half_train_dataset,
        shuffle=True,
        **loader_args)

    test_loader = torch.utils.data.DataLoader(
        dataset=original_test_dataset,
        shuffle=False,
        **loader_args)

    return {"train_loader": train_loader,
            "test_loader": test_loader}


dataset_factories = {
    'cifar10': get_cifar10_loaders,
    'half_cifar10': get_cifar10_half_loaders
}


def get_available_datasets():
    return dataset_factories.keys()


def get_dataset(name, *args, **kwargs):
    return dataset_factories[name](*args, **kwargs)