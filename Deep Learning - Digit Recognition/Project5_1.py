'''
Sumegha Singhania
Project5_1.py: corresponds to Task 1: A-E 
which involves getting the data set, making network code repeatable,
building the network model, training the model and saving it
The corresponding tasks have been mentioned as comments in their 
respective functions
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


# Global Variables:
n_epochs = 5
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

# class definitions

#1C: Build a network model
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

#function to train the network
def train(epoch,network,train_loader,optimizer,train_losses,train_counter):
	network.train()
	for batch_idx, (data, target) in enumerate(train_loader):
		optimizer.zero_grad()
		output = network(data)
		loss = F.nll_loss(output, target)
		loss.backward()
		optimizer.step()
		if batch_idx % log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.item()))
			train_losses.append(loss.item())
			train_counter.append(
				(batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
		#1E: Save the network to a file
		torch.save(network.state_dict(), 'model.pth')
		torch.save(optimizer.state_dict(), 'optimizer.pth')
	return

#function to test the network. It prints out the losses and accuracy of the network
def test(network,test_loader,test_losses):
	network.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for data, target in test_loader:
			output = network(data)
			test_loss += F.nll_loss(output, target, size_average=False).item()
			pred = output.data.max(1, keepdim=True)[1]
			correct += pred.eq(target.data.view_as(pred)).sum()
	test_loss /= len(test_loader.dataset)
	test_losses.append(test_loss)
	print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset),
		100. * correct / len(test_loader.dataset)))
	return



# main function: carries the execution flow of the network. All the predefined classes and 
#functions are used here
def main(argv):

	#1A: Get the MNIST digit data set
	train_loader = torch.utils.data.DataLoader(
  		torchvision.datasets.MNIST(root="data",train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  		batch_size=batch_size_train, shuffle=True)

	test_loader = torch.utils.data.DataLoader(
  		torchvision.datasets.MNIST(root="data",train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
 		batch_size=batch_size_test, shuffle=True)  

	#enumerate the testing data set
	examples = enumerate(test_loader)
	batch_idx, (example_data, example_targets) = next(examples)
	example_data.shape

	#plot a few examples and their corresponding ground truths
	fig = plt.figure()
	for i in range(6):
	  plt.subplot(2,3,i+1)
	  plt.tight_layout()
	  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
	  plt.title("Ground Truth: {}".format(example_targets[i]))
	  plt.xticks([])
	  plt.yticks([])
	fig  

#####################################################################################################

	#1B: Make your network code repeatable
	random_seed = 1
	torch.backends.cudnn.enabled = False
	torch.manual_seed(random_seed)

#####################################################################################################
	
	#1C: Build a network model
	network = Net()
	optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)
	#prints the structure of the network
	print(network)

#####################################################################################################

	#1D: Train the model

	#lists for keeping track of traversed data
	train_losses = []
	train_counter = []
	test_losses = []
	test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

	#to see the performance of the network without training
	test(network,test_loader,test_losses)

	#training and testing the performance after each epoch
	for epoch in range(1, n_epochs + 1):
		train(epoch,network,train_loader,optimizer,train_losses,train_counter)
		test(network,test_loader,test_losses)

	#plot for testing and training error
	fig = plt.figure()
	plt.plot(train_counter, train_losses, color='blue')
	plt.scatter(test_counter, test_losses, color='red')
	plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
	plt.xlabel('number of training examples seen')
	plt.ylabel('negative log likelihood loss')
	plt.show()
	fig

#####################################################################################################

	return

#to execute the script
if __name__ == "__main__":
    main(sys.argv)
