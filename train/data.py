import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, utils, datasets
from data_utils import SynData, combine_dataloaders
import os

class TwoCropTransformAdv:
    """Create two crops of the same image"""

    def __init__(self, transform, transform_adv):
        self.transform = transform
        self.transform_adv = transform_adv

    def __call__(self, x):
        return [self.transform(x), self.transform(x), self.transform_adv(x)]

def cifar10(data_dir, batch_size, use_syn=False, syn_data_path=None, batch_size_syn=0, contrastive=False):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    num_classes = 10
    transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), 
          transforms.ToTensor()])
    if contrastive:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ])
        transform = TwoCropTransformAdv(transform_train, transform)

    test_transform = transforms.ToTensor()

    train_data = datasets.CIFAR10(root=data_dir, train=True, transform=transform, download=True)
    if use_syn:
        # assuming large amount of synthetic data so additional augmentation unnecessary
        syn_data = SynData(syn_data_path, transform=test_transform)
        syn_loader = DataLoader(dataset=syn_data, batch_size=batch_size_syn, shuffle=True, num_workers=4)

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    if use_syn:
        train_loader = combine_dataloaders(train_loader, syn_loader)
    test_data = datasets.CIFAR10(root=data_dir, train=False, transform=test_transform, download=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader, mean, std, num_classes

def imagenette(data_dir, batch_size, use_syn=False, syn_data_path=None, batch_size_syn=0, size=224, contrastive=False):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    num_classes = 10
    
    transform_train = transforms.Compose([transforms.RandomResizedCrop(size),
                       transforms.RandomHorizontalFlip(),
                       transforms.ToTensor()
                       ])
    transform_test = transforms.Compose([transforms.Resize(int(1.14*size)),
                      transforms.CenterCrop(size),
                      transforms.ToTensor()
                      ])
    
    if contrastive:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ])
        transform_train = TwoCropTransformAdv(transform, transform_train)

    trainset = datasets.ImageFolder(
        os.path.join(data_dir, "train"), 
        transform=transform_train)
    testset = datasets.ImageFolder(
        os.path.join(data_dir, "val"), 
        transform=transform_test)
    
    if use_syn:
        # assuming large amount of synthetic data so additional augmentation unnecessary
        syn_data = SynData(syn_data_path, transform=transform_test)  
        syn_loader = DataLoader(dataset=syn_data, batch_size=batch_size_syn, shuffle=True, num_workers=4)

    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    if use_syn:
        train_loader = combine_dataloaders(train_loader, syn_loader)
        
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    
    return train_loader, test_loader, mean, std, num_classes

def imagenet100(data_dir, batch_size, use_syn=False, syn_data_path=None, batch_size_syn=0, size=224, contrastive=False):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    num_classes = 100

    transform_train = transforms.Compose([transforms.RandomResizedCrop(size),
                       transforms.RandomHorizontalFlip(),
                       transforms.ToTensor()
                       ])
    transform_test = transforms.Compose([transforms.Resize(int(1.14*size)),
                      transforms.CenterCrop(size),
                      transforms.ToTensor()
                      ])

    if contrastive:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ])
        transform_train = TwoCropTransformAdv(transform, transform_train)

    trainset = datasets.ImageFolder(
        os.path.join(data_dir, "train"),
        transform=transform_train)
    testset = datasets.ImageFolder(
        os.path.join(data_dir, "val"),
        transform=transform_test)

    if use_syn:
        # assuming large amount of synthetic data so additional augmentation unnecessary
        syn_data = SynData(syn_data_path, transform=transform_test)
        syn_loader = DataLoader(dataset=syn_data, batch_size=batch_size_syn, shuffle=True, num_workers=4)

    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    if use_syn:
        train_loader = combine_dataloaders(train_loader, syn_loader)

    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)

    return train_loader, test_loader, mean, std, num_classes


def cifar100(data_dir, batch_size, use_syn=False, syn_data_path=None, batch_size_syn=0, contrastive=False):
    mean = (0.507, 0.487, 0.441)
    std = (0.267, 0.256, 0.276)
    num_classes = 100
    transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), 
          transforms.ToTensor()])
    test_transform = transforms.ToTensor()

    if contrastive:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ])
        transform = TwoCropTransformAdv(transform_train, transform)

    train_data = datasets.CIFAR100(root=data_dir, train=True, transform=transform, download=True)
    if use_syn:
        # assuming large amount of synthetic data so additional augmentation unnecessary
        syn_data = SynData(syn_data_path, transform=test_transform)
        syn_loader = DataLoader(dataset=syn_data, batch_size=batch_size_syn, shuffle=True, num_workers=4)

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    if use_syn:
        train_loader = combine_dataloaders(train_loader, syn_loader)
    test_data = datasets.CIFAR100(root=data_dir, train=False, transform=test_transform, download=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader, mean, std, num_classes
