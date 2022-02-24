#!/usr/bin/env python
# coding: utf-8

# This script generates the MINICIFAR dataset from CIFAR10
# The following parameters can be changed : 
# n_classes (between 2 and 10) 
# Reduction factor R (which will result in 10000 /  R examples per class for the train set, and 1000 / R per class for test)
# --


train_size = 0.8

# Download the entire CIFAR10 dataset

from torchvision.datasets import CIFAR10
import numpy as np 
from torch.utils.data import Subset
from torch.utils.data.sampler import SubsetRandomSampler


import torchvision.transforms as transforms

## Normalization is different when training from scratch and when training using an imagenet pretrained backbone

normalize_scratch = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

# Data augmentation is needed in order to train from scratch
transform_train = transforms.Compose([
    transforms.RandomCrop(128, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize_scratch,
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize_scratch,
])


### The data from CIFAR10 will be downloaded in the following dataset
rootdir = './data/cifar10'

c10train = CIFAR10(rootdir,train=True,download=True,transform=transform_train)
c10test = CIFAR10(rootdir,train=False,download=True,transform=transform_test)

# CIFAR10 is sufficiently large so that training a model up to the state of the art performance will take approximately 3 hours on the 1060 GPU available on your machine. 
# As a result, we will create a "MiniCifar" dataset, based on CIFAR10, with less classes and exemples. 

def train_validation_split(train_size, num_train_examples):
    # obtain training indices that will be used for validation
    indices = list(range(num_train_examples))
    np.random.RandomState(seed=69).shuffle(indices)
    idx_split = int(np.floor(train_size * num_train_examples))
    train_index, valid_index = indices[:idx_split], indices[idx_split:]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_index)
    valid_sampler = SubsetRandomSampler(valid_index)

    return train_sampler,valid_sampler


### These dataloader are ready to be used to train for scratch 
num_train_examples=len(c10train)
train_sampler,valid_sampler=train_validation_split(train_size, num_train_examples)



from torch.utils.data.dataloader import DataLoader
trainloader = DataLoader(c10train,batch_size=128,sampler=train_sampler) 
validloader = DataLoader(c10train,batch_size=100,sampler=valid_sampler)
testloader = DataLoader(c10test,batch_size=100) 



#https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
#https://github.com/kentaroy47/vision-transformers-cifar10/blob/main/train_cifar10.py
