import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import glob
import os
import random

mean = {
    'cifar10': (0.4914, 0.4822, 0.4465)
}

std = {
    'cifar10': (0.2023, 0.1994, 0.2010)
}

transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean['cifar10'], std['cifar10'])])

def trainLoader(path):
    trainset = torchvision.datasets.CIFAR10(root=path, train=True,
                                            download=True, transform=transform)
    trainset.train_labels = [(label>0) for label in trainset.train_labels]
    indices_1 = [i for i,label in enumerate(trainset.train_labels) if label==1]
    count_1 = len(indices_1)
    indices_0 = [i for i,label in enumerate(trainset.train_labels) if label==0]
    count_0 = len(indices_0)
    train_data_0 = [trainset.train_data[i] for i in indices_0]
    print "Train Data 0 {}".format(len(train_data_0))
    random.shuffle(indices_1)
    train_data_1 = [trainset.train_data[i] for i in indices_1[:count_1/9]]
    print "Train Data 1 {}".format(len(train_data_1))
    train_labels_0 = [0 for i in indices_0]
    train_labels_1 = [1 for i in indices_1[:count_1/9]]
    print "Train Labels 0 {}".format(len(train_labels_0))
    print "Train Labels 1 {}".format(len(train_labels_1))

    train_labels = train_labels_0 + train_labels_1
    print "Train Labels {}".format(len(train_labels))
    train_data = train_data_0 + train_data_1
    print "Train Data {}".format(len(train_data))

    trainset.train_labels = train_labels
    trainset.train_data = train_data

    print trainset
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=4)
    return trainloader

def testLoader(path):
    testset = torchvision.datasets.CIFAR10(root=path, train=False,
                                           download=True, transform=transform)
    # print testset
    testset.test_labels = [(label>0) for label in testset.test_labels]
    indices_1 = [i for i,label in enumerate(testset.test_labels) if label==1]
    count_1 = len(indices_1)
    indices_0 = [i for i,label in enumerate(testset.test_labels) if label==0]
    count_0 = len(indices_0)
    test_data_0 = [testset.test_data[i] for i in indices_0]
    print "Train Data 0 {}".format(len(test_data_0))
    random.shuffle(indices_1)
    test_data_1 = [testset.test_data[i] for i in indices_1[:count_1/9]]
    print "Train Data 1 {}".format(len(test_data_1))
    test_labels_0 = [0 for i in indices_0]
    test_labels_1 = [1 for i in indices_1[:count_1/9]]
    print "Train Labels 0 {}".format(len(test_labels_0))
    print "Train Labels 1 {}".format(len(test_labels_1))

    test_labels = test_labels_0 + test_labels_1
    print "Train Labels {}".format(len(test_labels))
    test_data = test_data_0 + test_data_1
    print "Train Data {}".format(len(test_data))

    testset.test_labels = test_labels
    testset.test_data = test_data

    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False, num_workers=4)
    return testloader

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# def loader(path, batch_size=32, num_workers=4, pin_memory=True):
#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     return data.DataLoader(
#         datasets.ImageFolder(path,
#                              transforms.Compose([
#                                  transforms.Scale(256),
#                                  transforms.RandomSizedCrop(224),
#                                  transforms.RandomHorizontalFlip(),
#                                  transforms.ToTensor(),
#                                  normalize,
#                              ])),
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=num_workers,
#         pin_memory=pin_memory)
#
# def test_loader(path, batch_size=32, num_workers=4, pin_memory=True):
#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     return data.DataLoader(
#         datasets.ImageFolder(path,
#                              transforms.Compose([
#                                  transforms.Scale(256),
#                                  transforms.CenterCrop(224),
#                                  transforms.ToTensor(),
#                                  normalize,
#                              ])),
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=pin_memory)
