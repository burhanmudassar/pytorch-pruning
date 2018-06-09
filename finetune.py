import torch
from torch.autograd import Variable
from torchvision import models
import cv2
import sys
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dataset
from prune import *
import argparse
from operator import itemgetter
from heapq import nsmallest
import time

class ModifiedResNet101Model(torch.nn.Module):
	def __init__(self):
		super(ModifiedResNet101Model, self).__init__()

		model = models.resnet101(pretrained=True)
		self.conv1 = model.conv
		self.bn1 = model.bn1
		self.relu = model.relu
		self.maxpool = model.maxpool

		self.layer1 = model.layer1
		self.layer2 = model.layer2
		self.layer3 = model.layer3
		self.layer4 = model.layer4

		self.avgpool = model.avgpool

		for param in self.conv1.parameters():
			param.requires_grad = False
		for param in self.bn1.parameters():
			param.requires_grad = False
		for param in self.relu.parameters():
			param.requires_grad = False
		for param in self.maxpool.parameters():
			param.requires_grad = False

		for param in self.layer1.parameters():
			param.requires_grad = False
		for param in self.layer2.parameters():
			param.requires_grad = False
		for param in self.layer3.parameters():
			param.requires_grad = False
		for param in self.layer4.parameters():
			param.requires_grad = False

		for param in self.avgpool.parameters():
			param.requires_grad = False

		self.features = [self.conv1, self.bn1, self.relu, self.maxpool, self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool]

		# Create New Classifier
		self.fc = nn.Linear(512 * 4, 2)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		return x

class FilterPrunner:
	def __init__(self, model):
		self.model = model
		self.reset()
	
	def reset(self):
		# self.activations = []
		# self.gradients = []
		# self.grad_index = 0
		# self.activation_to_layer = {}
		self.filter_ranks = {}

	def forward(self, x):
		self.activations = []
		self.gradients = []
		self.grad_index = 0
		self.activation_to_layer = {}

		activation_index = 0
		for block in self.model.features:
			for layer, (name, module) in enumerate(block._modules.items()):
				x = module(x)
				if isinstance(module, torch.nn.modules.conv.Conv2d):
					x.register_hook(self.compute_rank)
					self.activations.append(x)
					self.activation_to_layer[activation_index] = layer
					activation_index += 1

		return self.model.fc(x.view(x.size(0), -1))

	def compute_rank(self, grad):
		activation_index = len(self.activations) - self.grad_index - 1
		activation = self.activations[activation_index]
		values = \
			torch.sum((activation * grad), dim = 0).\
				sum(dim=2).sum(dim=3)[0, :, 0, 0].data
		
		# Normalize the rank by the filter dimensions
		values = \
			values / (activation.size(0) * activation.size(2) * activation.size(3))

		if activation_index not in self.filter_ranks:
			self.filter_ranks[activation_index] = \
				torch.FloatTensor(activation.size(1)).zero_().cuda()

		self.filter_ranks[activation_index] += values
		self.grad_index += 1

	def lowest_ranking_filters(self, num):
		data = []
		for i in sorted(self.filter_ranks.keys()):
			for j in range(self.filter_ranks[i].size(0)):
				data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))

		return nsmallest(num, data, itemgetter(2))

	def normalize_ranks_per_layer(self):
		for i in self.filter_ranks:
			v = torch.abs(self.filter_ranks[i])
			v = v / np.sqrt(torch.sum(v * v))
			self.filter_ranks[i] = v.cpu()

	def get_prunning_plan(self, num_filters_to_prune):
		filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)

		# After each of the k filters are prunned,
		# the filter index of the next filters change since the model is smaller.
		filters_to_prune_per_layer = {}
		for (l, f, _) in filters_to_prune:
			if l not in filters_to_prune_per_layer:
				filters_to_prune_per_layer[l] = []
			filters_to_prune_per_layer[l].append(f)

		for l in filters_to_prune_per_layer:
			filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])
			for i in range(len(filters_to_prune_per_layer[l])):
				filters_to_prune_per_layer[l][i] = filters_to_prune_per_layer[l][i] - i

		filters_to_prune = []
		for l in filters_to_prune_per_layer:
			for i in filters_to_prune_per_layer[l]:
				filters_to_prune.append((l, i))

		return filters_to_prune				

class PrunningFineTuner_ResNet101:
	def __init__(self, train_path, test_path, model):
		# Insert CIFAR10 Loader
		# self.train_data_loader = dataset.loader(train_path)
		# self.test_data_loader = dataset.test_loader(test_path)

		self.model = model
		self.criterion = torch.nn.CrossEntropyLoss()
		self.prunner = FilterPrunner(self.model) 
		self.model.train()

	def test(self):
		self.model.eval()
		correct = 0
		total = 0

		for i, (batch, label) in enumerate(self.test_data_loader):
			batch = batch.cuda()
			output = model(Variable(batch))
			pred = output.data.max(1)[1]
	 		correct += pred.cpu().eq(label).sum()
	 		total += label.size(0)
	 	
	 	print "Accuracy :", float(correct) / total
	 	
	 	self.model.train()

	def train(self, optimizer = None, epoches = 10):
		if optimizer is None:
			optimizer = \
				optim.SGD(model.fc.parameters(),
					lr=0.0001, momentum=0.9)

		for i in range(epoches):
			print "Epoch: ", i
			self.train_epoch(optimizer)
			self.test()
		print "Finished fine tuning."
		

	def train_batch(self, optimizer, batch, label, rank_filters):
		self.model.zero_grad()
		input = Variable(batch)

		if rank_filters:
			output = self.prunner.forward(input)
			self.criterion(output, Variable(label)).backward()
		else:
			self.criterion(self.model(input), Variable(label)).backward()
			optimizer.step()

	def train_epoch(self, optimizer = None, rank_filters = False):
		for batch, label in self.train_data_loader:
			self.train_batch(optimizer, batch.cuda(), label.cuda(), rank_filters)

	def get_candidates_to_prune(self, num_filters_to_prune):
		self.prunner.reset()

		self.train_epoch(rank_filters = True)
		
		self.prunner.normalize_ranks_per_layer()

		return self.prunner.get_prunning_plan(num_filters_to_prune)
		
	def total_num_filters(self):
		filters = 0
		for blocks in self.model.features:
			for name, module in blocks._modules.items():
				if isinstance(module, torch.nn.modules.conv.Conv2d):
					filters = filters + module.out_channels
		return filters

	def prune(self):
		#Get the accuracy before prunning
		self.test()

		self.model.train()

		#Make sure all the layers are trainable
		for blocks in self.model.features:
			for param in blocks.parameters():
				param.requires_grad = True

		number_of_filters = self.total_num_filters()
		num_filters_to_prune_per_iteration = 512
		iterations = int(float(number_of_filters) / num_filters_to_prune_per_iteration)

		iterations = int(iterations * 2.0 / 3)

		print "Number of prunning iterations to reduce 67% filters", iterations

		for _ in range(iterations):
			print "Ranking filters.. "
			prune_targets = self.get_candidates_to_prune(num_filters_to_prune_per_iteration)
			layers_prunned = {}
			for layer_index, filter_index in prune_targets:
				if layer_index not in layers_prunned:
					layers_prunned[layer_index] = 0
				layers_prunned[layer_index] = layers_prunned[layer_index] + 1 

			print "Layers that will be prunned", layers_prunned
			print "Prunning filters.. "
			model = self.model.cpu()
			for layer_index, filter_index in prune_targets:
				model = prune_resnet101_conv_layer(model, layer_index, filter_index)

			self.model = model.cuda()

			message = str(100*float(self.total_num_filters()) / number_of_filters) + "%"
			print "Filters prunned", str(message)
			self.test()
			print "Fine tuning to recover from prunning iteration."
			optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
			self.train(optimizer, epoches = 10)


		print "Finished. Going to fine tune the model a bit more"
		self.train(optimizer, epoches = 15)
		torch.save(model, "model_prunned")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--prune", dest="prune", action="store_true")
    parser.add_argument("--train_path", type = str, default = "train")
    parser.add_argument("--test_path", type = str, default = "test")
    parser.set_defaults(train=False)
    parser.set_defaults(prune=False)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
	args = get_args()

	if args.train:
		model = ModifiedResNet101Model().cuda()
	elif args.prune:
		model = torch.load("model").cuda()

	fine_tuner = PrunningFineTuner_ResNet101(args.train_path, args.test_path, model)

	if args.train:
		fine_tuner.train(epoches = 20)
		torch.save(model, "model")

	elif args.prune:
		fine_tuner.prune()
