import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import glob
import os

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def trainLoader(path):
    trainset = torchvision.datasets.CIFAR10(root=path, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=4)
    return trainloader

def testLoader(path):
    testset = torchvision.datasets.CIFAR10(root=path, train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
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