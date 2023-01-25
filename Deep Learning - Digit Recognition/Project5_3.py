'''
Sumegha Singhania
Project5_3: Corresponds to task 3, which involves,
creating a digital embedding space using our network trained so 
far as the embedding spacr for images of written greek symbols
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
import pandas as pd
import csv

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


#3B: Create a truncated model that terminates at the Dense layer with 50 outputs
class Submodel(Net):
	def __init__(self):
		Net.__init__(self)

	def forward(self, x):
		x = F.relu( F.max_pool2d( self.conv1(x), 2 ) ) # relu on max pooled results of conv1
		x = F.relu( F.max_pool2d( self.conv2_drop( self.conv2(x)), 2 ) ) # relu on max pooled results of dropout of conv2
		x = x.view(-1, 320)
		x = F.relu(self.fc1(x))
		x = F.dropout(x, training=self.training)
		return 

# useful functions with a comment for each function


#3A: Create a greek symbol data set
#function to read the greek data sybmols and save them to a csv file
def read_data(data_dir,batch,csv_path):
	
	f1 = open(csv_path + '/greek_intensity.csv', 'w')
	f1.write("Greek Symbol Intensities \n")

	f2 = open(csv_path + '/greek_labels.csv', 'w')
	f2.write('Greek Labels \n')

	
	conv_tens = torchvision.transforms.Compose([
	                               torchvision.transforms.ToTensor(),
	                               torchvision.transforms.Resize((28,28)),
	                               torchvision.transforms.Normalize(
	                                 (0.1307,), (0.3081,)),
	                               torchvision.transforms.Grayscale()     
	                             ])

	i=0;
	for filename in os.listdir(data_dir):
		if(filename!='.DS_Store'):
			img = Image.open(os.path.join(data_dir,filename)).convert('RGB')
			if img is not None:
				temp = conv_tens(img)
				temp = torchvision.transforms.functional.invert(temp)
				batch[i] = temp
				i=i+1
				s=""
				# print(temp.size())
				for row in range(27):
					for column in range(27):
						val = temp[0][row][column]
						s=s+numpy.array2string(val.numpy())+","
				f1.write(s)
				f1.write("\n")


				if filename.split("_")[0] == "alpha":
					f2.write("0" + "\n")
				if filename.split("_")[0] == "beta":
					f2.write("1" + "\n")
				if filename.split("_")[0] == "gamma":
					f2.write("2" + "\n")
	
	f1.close()
	f2.close()

	return



# def load_data(csv_path):
# 	# greek_int_data):
# 	path = csv_path + '/greek_intensity.csv'
	
# 	load_val = []

# 	with open(csv_path + '/greek_intensity.csv') as f1:
# 		header = next(f1)
# 		reader = csv.reader(f1)   
# 		load_val = list(reader)
		
	
	# tmp = numpy.array(float(load_val))
	# tmp = torch.from_numpy(tmp)
	# batch_size = len(load_val)
	# batch_data = torch.zeros(batch_size, 1, 28, 28, dtype=torch.float32)

	# for i in range(len(load_val)):
	# 	k=0
	# 	tmp = torch.zeros(len(load_val[i]), 1, 28, 28, dtype=torch.float64)
	# 	tmp = []
	# 	for j in range(len(load_val[i])):
	# 		s = load_val[i][j]
	# 		if s!='':
	# # 			# print(float(s))
				# tmp.append(float(s))
	# 			tmp[k] = float(s)
	# 			k=k+1
	# 			# batch_data[0][i][k]=float(s)
	# 			# k=k+1
	# 	print("next row")
	# 	print(tmp)
		# print(tmp)
		# tmp = torch.Tensor(tmp)
		# batch_data[i]=tmp
			# s.split(",")
			# print(len(s))
			# for k in range(len(s)):
			# 	# val = int(float(s[k]))
			# 	# print(val)
			# 	print(s[k])



	# return load_val


# carries the execution flow of the network. All the predefined classes and 
#functions are used here
def main(argv):

	#load the trained model
	model = Net()
	model.load_state_dict(torch.load("model.pth"))
	model.eval()
	# print("Trained model: ")
	print(model)

#####################################################################################################

	#3A: Create a greek symbol data set
	data_dir = "/Users/sumeghasinghania/Desktop/CV_Projects/Project5/greek-1"
	csv_path = "/Users/sumeghasinghania/Desktop/CV_Projects/Project5/CV_5"
	filenames = [name for name in os.listdir(data_dir) if os.path.splitext(name)[-1] == '.png']
	batch_size = len(filenames)
	batch = torch.zeros(batch_size, 1, 28, 28, dtype=torch.float32)
	read_data(data_dir,batch,csv_path)

#####################################################################################################

	#3B: Create a truncated model
	model_truncated = Submodel()
	# print("Truncated model: ")
	print(model_truncated)
	model_truncated.load_state_dict(torch.load("model.pth"))
	model_truncated.eval()

	#load the test data
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

	#applying the truncated model to the test data
	with torch.no_grad():
		output = model_truncated(example_data)

	# print(output.size())

	#####################################################################################################

	#3C: Project the greek symbols into the embedding space
	# apply the truncated model to the greek symbol data we loaded
	with torch.no_grad():
		output_2 = model_truncated(batch)

	# plot the greek symbols and the model predictions 
	fig = plt.figure()
	for i in range(27):
	  plt.subplot(9,3,i+1)
	  plt.tight_layout()
	  plt.imshow(batch[i][0], cmap='gray', interpolation='none')
	  plt.title("Prediction: {}".format(
	    output_2.data.max(1, keepdim=True)[1][i].item()))
	  plt.xticks([])
	  plt.yticks([])
	plt.show()
	fig

#####################################################################################################

	#3D: Compute distances in the embedding space

	#to store the ssd results
	ssd_alpha = []
	ssd_beta = []
	ssd_gamma = []

	#gamma
	for i in range(27):
		ssd = numpy.sum((numpy.array(output[0])-numpy.array(output[i]))**2)
		ssd_gamma.append(ssd)

	#alpha
	for i in range(27):
		ssd = numpy.sum((numpy.array(output[7])-numpy.array(output[i]))**2)
		ssd_alpha.append(ssd)

	#beta
	for i in range(27):
		ssd = numpy.sum((numpy.array(output[8])-numpy.array(output[i]))**2)
		ssd_beta.append(ssd)

	#print the ssd lists
	print(ssd_gamma)
	print(ssd_alpha)
	print(ssd_beta)

#####################################################################################################

	#3E: Create your own greek symbol data
	#loading handwritten greek symbols
	data_dir = '/Users/sumeghasinghania/Desktop/CV_Projects/Project5/greek_self/'
	filenames = [name for name in os.listdir(data_dir) if os.path.splitext(name)[-1] == '.jpg']

	batch_size = len(filenames)
	batch = torch.zeros(batch_size, 1, 28, 28, dtype=torch.float32)

	conv_tens = torchvision.transforms.Compose([
	                               torchvision.transforms.ToTensor(),
	                               torchvision.transforms.Resize((28,28)),
	                               torchvision.transforms.Normalize(
	                                 (0.1307,), (0.3081,)),
	                               torchvision.transforms.Grayscale()
	                             ])

	#converting the input data to a tensor
	i=0;
	for filename in os.listdir(data_dir):
		if(filename!='.DS_Store'):
			img = Image.open(os.path.join(data_dir,filename))
			if img is not None:
				temp = conv_tens(img)
				temp = torchvision.transforms.functional.invert(temp)
				batch[i] = temp
				i=i+1

	batch.shape

	#applying the model to our tensor/input data
	with torch.no_grad():
		output_3 = model_truncated(batch)

	#plot the input data and the model prediction
	fig = plt.figure()
	for i in range(3):
	  plt.subplot(1,3,i+1)
	  plt.tight_layout()
	  plt.imshow(batch[i][0], cmap='gray', interpolation='none')
	  plt.title("Prediction: {}".format(
	    output_3.data.max(1, keepdim=True)[1][i].item()))
	  plt.xticks([])
	  plt.yticks([])
	plt.show()
	fig

	return

# to execute the script
if __name__ == "__main__":
    main(sys.argv)
