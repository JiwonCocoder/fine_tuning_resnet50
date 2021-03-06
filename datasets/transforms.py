import pdb
import random
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler

from .augmentation.randaugment import RandAugment
from .data_utils import get_sampler_by_name, get_data_loader, get_onehot, split_ssl_data
from .dataset import BasicDataset

import torchvision
from torchvision import datasets, transforms

mean, std = {}, {}
mean['cifar10'] = [x / 255 for x in [125.3, 123.0, 113.9]]
mean['cifar100'] = [x / 255 for x in [129.3, 124.1, 112.4]]
mean['MLCC'] = [0.1778, 0.04714, 0.16583]
mean['SVHN'] = [0.4380, 0.4440, 0.4730]
mean['STL10'] = [0.4467, 0.4398, 0.4066]

std['cifar10'] = [x / 255 for x in [63.0, 62.1, 66.7]]
std['cifar100'] = [x / 255 for x in [68.2,  65.4,  70.4]]
std['MLCC'] = [0.26870, 0.1002249, 0.273526]
std['SVHN'] = [0.1751, 0.1771, 0.1744]
std['STL10'] = [0.2241, 0.2215, 0.2239]

def get_transform(name, learning_type, train):
    if not train:
        if name == 'MLCC':
            return transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize(mean[name], std[name])])
        else:
            return transforms.Compose([transforms.Resize(size=(224,224)),
                                        transforms.ToTensor(), transforms.Normalize(mean[name], std[name])])
    elif train:
        if name == 'MLCC':
            data_transforms = transforms.Compose([
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(mean[name], std[name])
                ])
        else:
            data_transforms = transforms.Compose([
                    transforms.Resize(size=(224,224)),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(mean[name], std[name])
                ])
    return data_transforms

        # if learning_type == 'semi':
        #     data_transforms = transforms.Compose([
        #         transforms.RandomHorizontalFlip(p=0.5),
        #         transforms.RandomVerticalFlip(p=0.5),
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean[name], std[name])
        #     ])
        #     return data_transforms
        # elif learning_type == 'sup':
        #     data_transforms= transforms.Compose([
        #         transforms.RandomVerticalFlip(p=0.5),
        #         transforms.RandomHorizontalFlip(p=0.5),
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean[name], std[name])
        #     ])
        #     return data_transforms

