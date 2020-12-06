import os

import torch
from torch.utils.data import random_split, ConcatDataset
from torchvision import transforms, datasets
import  numpy as np

from functools import partial



def get_svhn_loaders(use_half_train=False):
    train_transform = transforms.Compose([transforms.ToTensor()])

    dataset = datasets.SVHN(root=os.path.join('data', 'svhn_data'),
                            transform=train_transform, download=True)
    val_dataset_size = int(len(dataset) / 3)
    train_dataset_size = len(dataset) - val_dataset_size
    train_dataset, val_dataset = random_split(dataset, [train_dataset_size, val_dataset_size])

    if use_half_train:
        dataset_size = len(train_dataset)
        split = int(np.floor(0.5 * dataset_size))
        train_dataset, _ = random_split(train_dataset, [dataset_size - split, split])

    loader_args = {
        "batch_size": 128,

    }

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        shuffle=True,
        **loader_args)

    test_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        shuffle=False,
        **loader_args)

    return {"train_loader": train_loader,
            "test_loader": test_loader}


def get_cifar10_loaders(use_half_train=False):
    normalize_transform = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_transform = transforms.Compose([transforms.ToTensor(), normalize_transform])
    test_transform = transforms.Compose([transforms.ToTensor(), normalize_transform])

    original_train_dataset = datasets.CIFAR10(root=os.path.join('data', 'cifar10_data'),
                                              train=True, transform=train_transform, download=True)
    original_test_dataset = datasets.CIFAR10(root=os.path.join('data', 'cifar10_data'),
                                             train=False, transform=test_transform, download=True)

    if use_half_train:
        dataset_size = len(original_train_dataset)
        split = int(np.floor(0.5 * dataset_size))
        original_train_dataset, _ = random_split(original_train_dataset, [dataset_size - split, split])

    loader_args = {
        "batch_size": 128,

    }

    train_loader = torch.utils.data.DataLoader(
        dataset=original_train_dataset,
        shuffle=True,
        **loader_args)

    test_loader = torch.utils.data.DataLoader(
        dataset=original_test_dataset,
        shuffle=False,
        **loader_args)

    return {"train_loader": train_loader,
            "test_loader": test_loader}


def get_cifar100_loaders(use_half_train=False):
    normalize_transform = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_transform = transforms.Compose([transforms.ToTensor(), normalize_transform])
    test_transform = transforms.Compose([transforms.ToTensor(), normalize_transform])

    original_train_dataset = datasets.CIFAR100(root=os.path.join('data', 'cifar100_data'),
                                              train=True, transform=train_transform, download=True)
    original_test_dataset = datasets.CIFAR100(root=os.path.join('data', 'cifar100_data'),
                                             train=False, transform=test_transform, download=True)

    if use_half_train:
        dataset_size = len(original_train_dataset)
        split = int(np.floor(0.5 * dataset_size))
        original_train_dataset, _ = random_split(original_train_dataset, [dataset_size - split, split])

    loader_args = {
        "batch_size": 128,

    }

    train_loader = torch.utils.data.DataLoader(
        dataset=original_train_dataset,
        shuffle=True,
        **loader_args)

    test_loader = torch.utils.data.DataLoader(
        dataset=original_test_dataset,
        shuffle=False,
        **loader_args)

    return {"train_loader": train_loader,
            "test_loader": test_loader}


def get_cifar10_partial_with_val_loader(n_train):
    normalize_transform = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_transform = transforms.Compose([transforms.ToTensor(), normalize_transform])
    test_transform = transforms.Compose([transforms.ToTensor(), normalize_transform])

    original_train_dataset = datasets.CIFAR10(root=os.path.join('data', 'cifar10_data'),
                                              train=True, transform=train_transform, download=True)
    original_test_dataset = datasets.CIFAR10(root=os.path.join('data', 'cifar10_data'),
                                             train=False, transform=test_transform, download=True)

    
    val_dataset_size = int(len(original_train_dataset) / 3)
    train_dataset_size = len(original_train_dataset) - val_dataset_size
    train_dataset, val_dataset = random_split(original_train_dataset, [train_dataset_size, val_dataset_size])
    train_dataset, _ = random_split(train_dataset, [n_train, train_dataset_size - n_train])

    loader_args = {
        "batch_size": 128,

    }

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        shuffle=True,
        **loader_args)

    test_loader = torch.utils.data.DataLoader(
        dataset=original_test_dataset,
        shuffle=False,
        **loader_args)
    
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        shuffle=False,
        **loader_args)
    
    return {"train_loader": train_loader,
        "test_loader": test_loader,
        "val_loader": val_loader}
    
def get_cifar10_online_with_val_loader(split_size):
    normalize_transform = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_transform = transforms.Compose([transforms.ToTensor(), normalize_transform])
    test_transform = transforms.Compose([transforms.ToTensor(), normalize_transform])

    original_train_dataset = datasets.CIFAR10(root=os.path.join('data', 'cifar10_data'),
                                              train=True, transform=train_transform, download=True)
    original_test_dataset = datasets.CIFAR10(root=os.path.join('data', 'cifar10_data'),
                                             train=False, transform=test_transform, download=True)

    
    val_dataset_size = int(len(original_train_dataset) / 3)
    train_dataset_size = len(original_train_dataset) - val_dataset_size
    original_train_dataset, val_dataset = random_split(original_train_dataset, [train_dataset_size, val_dataset_size])
    
    train_datasets = random_split(original_train_dataset, [split_size for _ in range(train_dataset_size // split_size)] + [train_dataset_size % split_size])

    loader_args = {
        "batch_size": 128,

    }

    train_loaders = []
    active_datasets = []
    for train_dataset in train_datasets:
        active_datasets.append(train_dataset)
        train_loaders.append(torch.utils.data.DataLoader(
            dataset=ConcatDataset(active_datasets),
            shuffle=True,
            **loader_args
        ))

    test_loader = torch.utils.data.DataLoader(
        dataset=original_test_dataset,
        shuffle=False,
        **loader_args)
    
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        shuffle=False,
        **loader_args)

    return {"train_loaders": train_loaders,
            "test_loader": test_loader,
            "val_loader": val_loader}




dataset_factories = {
    'cifar10': get_cifar10_loaders,
    'half_cifar10': partial(get_cifar10_loaders, use_half_train=True),
    'partial_with_val_cifar10': get_cifar10_partial_with_val_loader,
    'online_with_val_cifar10': get_cifar10_online_with_val_loader,
    'svhn': get_svhn_loaders,
    'half_svhn': partial(get_svhn_loaders, use_half_train=True),
    'cifar100': get_cifar100_loaders,
    'half_cifar100': partial(get_cifar100_loaders, use_half_train=True),
}


def get_available_datasets():
    return dataset_factories.keys()


def get_dataset(name, *args, **kwargs):
    return dataset_factories[name](*args, **kwargs)