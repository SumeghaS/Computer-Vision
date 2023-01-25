'''
Sumegha Singhania
Project5_1FG.py: correspononds to Task F,G, which involve
reading the trained network from the saved locarion, running it on a test set,
running it on new inputs of handwritten digits.
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

batch_size_train = 64
batch_size_test = 1000


# class definitions

#Build a network model
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


# main function: carries the execution flow of the network. All the predefined classes and 
#functions are used here
def main(argv):

	#1F: Read the network and run it on the test set
	model = Net()
	model.load_state_dict(torch.load("model.pth"))
	model.eval()

	#loading the test set
	training_data = torch.utils.data.DataLoader(
	  torchvision.datasets.MNIST(root="data",train=True, download=True,
	                             transform=torchvision.transforms.Compose([
	                               torchvision.transforms.ToTensor(),
	                               torchvision.transforms.Normalize(
	                                 (0.1307,), (0.3081,))
	                             ])),
	  batch_size=batch_size_train, shuffle=True)

	test_data = torch.utils.data.DataLoader(
	  torchvision.datasets.MNIST(root="data",train=False, download=True,
	                             transform=torchvision.transforms.Compose([
	                               torchvision.transforms.ToTensor(),
	                               torchvision.transforms.Normalize(
	                                 (0.1307,), (0.3081,))
	                             ])),
	batch_size=batch_size_test, shuffle=True)

	#enumereating the test set to run the model
	examples = enumerate(test_data)
	batch_idx, (example_data, example_targets) = next(examples)
	example_data.shape

	#applying the model to the test set
	with torch.no_grad():
	  output = model(example_data)

	#plotting the test set examples and model predictions
	fig = plt.figure()
	for i in range(9):
	  plt.subplot(3,3,i+1)
	  plt.tight_layout()
	  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
	  plt.title("Prediction: {}".format(
	    output.data.max(1, keepdim=True)[1][i].item()))
	  plt.xticks([])
	  plt.yticks([])
	# plt.show()
	fig

#####################################################################################################

	#1G: Test the network on new inputs

	#loading new inputs
	data_dir = '/Users/sumeghasinghania/Desktop/CV_Projects/Project5/data_numbers/'
	filenames = [name for name in os.listdir(data_dir) if os.path.splitext(name)[-1] == '.jpeg']

	batch_size = len(filenames)
	batch = torch.zeros(batch_size, 1, 28, 28, dtype=torch.float32)

	conv_tens = torchvision.transforms.Compose([
	                               torchvision.transforms.ToTensor(),
	                               torchvision.transforms.Resize((28,28)),
	                               torchvision.transforms.Normalize(
	                                 (0.1307,), (0.3081,)),
	                               torchvision.transforms.Grayscale()
	                             ])

	#converting the new inputs into a tensor - batch
	i=0;
	for filename in os.listdir(data_dir):
		if(filename!='.DS_Store'):
			img = Image.open(os.path.join(data_dir,filename))
			if img is not None:
				temp = conv_tens(img)
				batch[i] = temp
				i=i+1

	batch.shape

	#applying the model to the new input
	with torch.no_grad():
		output_2 = model(batch)

	#plotting the input and the model predictions
	fig = plt.figure()
	for i in range(9):
	  plt.subplot(3,3,i+1)
	  plt.tight_layout()
	  plt.imshow(batch[i][0], cmap='gray', interpolation='none')
	  plt.title("Prediction: {}".format(
	    output_2.data.max(1, keepdim=True)[1][i].item()))
	  plt.xticks([])
	  plt.yticks([])
	plt.show()
	fig
#####################################################################################################

	return

#to execute the script
if __name__ == "__main__":
	main(sys.argv)
