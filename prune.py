import torch
from torch.autograd import Variable
from torchvision import models
import cv2
import sys
import numpy as np
import pyjack
 
def replace_layers(model, i, indexes, layers):
	if i in indexes:
		return layers[indexes.index(i)]
	return model[i]

def prune_resnet101_conv_layer(model, layer_index, filter_index):
	#layer_to_prune = []
	#layers_affected = []
	layer_affected = []
	layer = 0
	activation_index = 0
	for block in model.features:
		if block == model.conv1 or block == model.bn1:
			# print "Adding initial conv1 or bn1 to graph"
			# print block
			if block == model.conv1:
				pass
				#print "conv1"
			else:
				pass
				#print "bn1"
			if isinstance(block, torch.nn.modules.conv.Conv2d):
				if layer_index == layer:
					layer_to_prune = block
					print model.layer1
					layer_affected.append(model.bn1)
					for (bottleneck_name, bottleneck) in model.layer1._modules.items():
						if bottleneck_name == '0':
							for (layer_name, module) in bottleneck._modules.items():
								if layer_name == 'conv1':
									layer_affected.append(module)
				activation_index += 1
				layer += 1
		elif block == model.layer1 or block == model.layer2 or block == model.layer3 or block == model.layer4:
			# print "Adding Residual Block to graph"
			for (block_name, bottleneck) in block._modules.items():
				# print block_name
				for (name, module) in bottleneck._modules.items():
					# print name,module
					if isinstance(module, torch.nn.modules.Sequential):
						pass
						#print "Sequential Block"
						#out += module(x)
						#x = F.relu(out)
					else:
						# print "Conv or BN Inside Bottleneck"
						# print name, module
						if name == 'conv1':
							pass
							#print "bottleneck conv1"
							#out = module(x)
						elif name == 'bn1' or name == 'bn2':
							pass
							#print "bottleneck bn1 or bn2"
							#out = F.relu(module(out))
						elif name == 'conv2' or name == 'conv3':
							pass
							#print "bottleneck conv2 or conv3"
							#out = module(out)
						elif name == 'bn3':
							pass
							#print "bottleneck bn3"
							#out = module(out)
						else:
							pass
							#out = module(out)
						if isinstance(module, torch.nn.modules.conv.Conv2d):
							if layer_index == layer:
								layer_to_prune = module
							activation_index += 1
							layer += 1

	conv = layer_to_prune
	print "Pruning This Conv Block"
	print conv
	print "Layers Affected"
	print layer_affected
	#next_conv = None
	#offset = 1

	# while layer_index + offset <  len(model.features._modules.items()):
	# 	res =  model.features._modules.items()[layer_index+offset]
	# 	if isinstance(res[1], torch.nn.modules.conv.Conv2d):
	# 		next_name, next_conv = res
	# 		break
	# 	offset = offset + 1
	#
	new_conv = \
		torch.nn.Conv2d(in_channels = conv.in_channels, \
			out_channels = conv.out_channels - 1,
			kernel_size = conv.kernel_size, \
			stride = conv.stride,
			padding = conv.padding,
			dilation = conv.dilation,
			groups = conv.groups,
			bias = conv.bias)

	old_weights = conv.weight.data.cpu().numpy()
	new_weights = new_conv.weight.data.cpu().numpy()

	new_weights[: filter_index, :, :, :] = old_weights[: filter_index, :, :, :]
	new_weights[filter_index : , :, :, :] = old_weights[filter_index + 1 :, :, :, :]
	new_conv.weight.data = torch.from_numpy(new_weights).cuda()
	if conv.bias:
		bias_numpy = conv.bias.data.cpu().numpy()

		bias = np.zeros(shape = (bias_numpy.shape[0] - 1), dtype = np.float32)
		bias[:filter_index] = bias_numpy[:filter_index]
		bias[filter_index : ] = bias_numpy[filter_index + 1 :]
		new_conv.bias.data = torch.from_numpy(bias).cuda()

	print "Adding new Conv"
	print new_conv
	
	conv = pyjack.replace_all_refs(conv,new_conv)

	new_in_channels = new_conv.out_channels	

	for layers in layer_affected:
		if isinstance(layers, torch.nn.modules.conv.Conv2d):
			new_layer = \
				torch.nn.Conv2d(in_channels = new_in_channels, \
				out_channels = layers.out_channels,
				kernel_size = layers.kernel_size, \
				stride = layers.stride,
				padding = layers.padding,
				dilation = layers.dilation,
				groups = layers.groups,
				bias = layers.bias)

			old_weights = layers.weight.data.cpu().numpy()
			new_weights = new_layer.weight.data.cpu().numpy()

			new_weights[:, : filter_index, :, :] = old_weights[:, : filter_index, :, :]
			new_weights[:, filter_index :, :, :] = old_weights[:, filter_index + 1 :, :, :]
			new_layer.weight.data = torch.from_numpy(new_weights).cuda()
			if layers.bias:
				bias_numpy = layers.bias.data.cpu().numpy()
				new_layer.bias.data = torch.from_numpy(bias_numpy).cuda()
		if isinstance(layers, torch.nn.modules.batchnorm.BatchNorm2d):
			new_layer = torch.nn.BatchNorm2d(new_in_channels)
			old_weights = layers.weight.data.cpu().numpy()
			

		new_in_channels = new_layer.out_channels
	
	layers = pyjack.replace_all_refs(layers,new_layer)

	new_in_channels = new_conv.out_channels	


	#print "Conv1 New"
	#print model.conv1

	return model

	# if not next_conv is None:
	# 	next_new_conv = \
	# 		torch.nn.Conv2d(in_channels = next_conv.in_channels - 1,\
	# 			out_channels =  next_conv.out_channels, \
	# 			kernel_size = next_conv.kernel_size, \
	# 			stride = next_conv.stride,
	# 			padding = next_conv.padding,
	# 			dilation = next_conv.dilation,
	# 			groups = next_conv.groups,
	# 			bias = next_conv.bias)
    #
	# 	old_weights = next_conv.weight.data.cpu().numpy()
	# 	new_weights = next_new_conv.weight.data.cpu().numpy()
    #
	# 	new_weights[:, : filter_index, :, :] = old_weights[:, : filter_index, :, :]
	# 	new_weights[:, filter_index : , :, :] = old_weights[:, filter_index + 1 :, :, :]
	# 	next_new_conv.weight.data = torch.from_numpy(new_weights).cuda()
    #
	# 	next_new_conv.bias.data = next_conv.bias.data

	# if not next_conv is None:
	#  	features = torch.nn.Sequential(
	#             *(replace_layers(model.features, i, [layer_index, layer_index+offset], \
	#             	[new_conv, next_new_conv]) for i, _ in enumerate(model.features)))
	#  	del model.features
	#  	del conv
    #
	#  	model.features = features
    #
	# else:
	# 	#Prunning the last conv layer. This affects the first linear layer of the classifier.
	#  	model.features = torch.nn.Sequential(
	#             *(replace_layers(model.features, i, [layer_index], \
	#             	[new_conv]) for i, _ in enumerate(model.features)))
	#  	layer_index = 0
	#  	old_linear_layer = None
	#  	for _, module in model.classifier._modules.items():
	#  		if isinstance(module, torch.nn.Linear):
	#  			old_linear_layer = module
	#  			break
	#  		layer_index = layer_index  + 1
    #
	#  	if old_linear_layer is None:
	#  		raise BaseException("No linear laye found in classifier")
	# 	params_per_input_channel = old_linear_layer.in_features / conv.out_channels
    #
	#  	new_linear_layer = \
	#  		torch.nn.Linear(old_linear_layer.in_features - params_per_input_channel,
	#  			old_linear_layer.out_features)
	#
	#  	old_weights = old_linear_layer.weight.data.cpu().numpy()
	#  	new_weights = new_linear_layer.weight.data.cpu().numpy()
    #
	#  	new_weights[:, : filter_index * params_per_input_channel] = \
	#  		old_weights[:, : filter_index * params_per_input_channel]
	#  	new_weights[:, filter_index * params_per_input_channel :] = \
	#  		old_weights[:, (filter_index + 1) * params_per_input_channel :]
	#
	#  	new_linear_layer.bias.data = old_linear_layer.bias.data
    #
	#  	new_linear_layer.weight.data = torch.from_numpy(new_weights).cuda()
    #
	# 	classifier = torch.nn.Sequential(
	# 		*(replace_layers(model.classifier, i, [layer_index], \
	# 			[new_linear_layer]) for i, _ in enumerate(model.classifier)))
    #
	# 	del model.classifier
	# 	del next_conv
	# 	del conv
	# 	model.classifier = classifier

if __name__ == '__main__':
	model = models.vgg16(pretrained=True)
	model.train()

	t0 = time.time()
	model = prune_conv_layer(model, 28, 10)
	print "The prunning took", time.time() - t0
