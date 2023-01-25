'''
Sumegha Singhania
Project5_2.py: corresponds to Task 2, which involves,
examining the network by analyzing the first convolution layer, 
plotting the effects of the filters, and building a truncated 
network model with just the first 2 convilution layers.
'''
#import statements
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import cv2
import sys
from PIL import Image
import os
import numpy

#global variables
batch_size_train = 64
batch_size_test = 1000


# class definitions

# Build a network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

# Build a submodel with only first 2 convolution layers
class Submodel(Net):
	def __init__(self):
		Net.__init__(self)

	def forward(self, x):
		x = F.relu( F.max_pool2d( self.conv1(x), 2 ) ) # relu on max pooled results of conv1
		x = F.relu( F.max_pool2d( self.conv2_drop( self.conv2(x)), 2 ) ) # relu on max pooled results of dropout of conv2
		return x

# main function: carries the execution flow of the network. All the predefined classes and 
#functions are used here
def main(argv):

	model = Net()
	model.load_state_dict(torch.load("model.pth"))
	model.eval()
	print(model)


	#2A: Analyze the first layer
	weight_tensor = model.conv1.weight
	# print(weight_tensor)

	#plot the first layer filters
	with torch.no_grad():
		fig = plt.figure()
		for i in range(10):
		  plt.subplot(3,4,i+1)
		  plt.tight_layout()
		  plt.imshow(weight_tensor[i][0], interpolation='none')
		  plt.title("Filter: {}".format(i))
		  plt.xticks([])
		  plt.yticks([])
		# plt.show()
		fig
#####################################################################################################
	
	#loading the test set data
	test_data = torch.utils.data.DataLoader(
	  torchvision.datasets.MNIST(root="data",train=False, download=True,
	                             transform=torchvision.transforms.Compose([
	                               torchvision.transforms.ToTensor(),
	                               torchvision.transforms.Normalize(
	                                 (0.1307,), (0.3081,))
	                             ])),
	batch_size=batch_size_test, shuffle=True)
	examples = enumerate(test_data)
	batch_idx, (example_data, example_targets) = next(examples)
	example_data.shape

	#2B: Show the effect of the filters
	#plot the effects of the filters on the data
	with torch.no_grad():
		j=1
		fig = plt.figure()
		for i in range(10):
		  plt.subplot(5,4,j)
		  plt.tight_layout()
		  plt.imshow(weight_tensor[i][0], cmap='gray', interpolation='none')
		  plt.title("Filter: {}".format(i))
		  plt.xticks([])
		  plt.yticks([])
		  j=j+1
		  plt.subplot(5,4,j)
		  #Using opencv filter2D to apply filters to the example
		  img = cv2.filter2D(example_data[0][0].detach().numpy(),-1,weight_tensor[i][0].detach().numpy())
		  plt.imshow(img, cmap='gray', interpolation='none')
		  j=j+1
		  plt.xticks([])
		  plt.yticks([])
		# plt.show()
		fig

#####################################################################################################

	#2C: Build a truncated model
	#calling the submodel we created
	model_truncated = Submodel()
	# print(model_truncated)
	model_truncated.load_state_dict(torch.load("model.pth"))
	model_truncated.eval()

	#applying the first model to our test data
	with torch.no_grad():
		output = model_truncated(example_data)

	#checking if the output has 20 4x4 channels
	print(output.size())

	#weight tensor after the seconf convolution layer
	weight_tensor_2 = model_truncated.conv2.weight

	#plot the effects of the filters on the data
	with torch.no_grad():
		j=1
		fig = plt.figure()
		for i in range(10):
		  plt.subplot(5,4,j)
		  plt.tight_layout()
		  plt.imshow(weight_tensor_2[i][0], cmap='gray', interpolation='none')
		  plt.title("Filter: {}".format(i))
		  plt.xticks([])
		  plt.yticks([])
		  j=j+1
		  plt.subplot(5,4,j)
		  #applying the effects of the filters after the second convolution layer
		  img = cv2.filter2D(example_data[0][0].detach().numpy(),-1,weight_tensor_2[i][0].detach().numpy())
		  plt.imshow(img, cmap='gray', interpolation='none')
		  j=j+1
		  plt.xticks([])
		  plt.yticks([])
		plt.show()
	return

#to execute the script
if __name__ == "__main__":
	main(sys.argv)