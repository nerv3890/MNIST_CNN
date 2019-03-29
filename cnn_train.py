import torch
import torchvision
import torch.nn.functional as F

EPOCH = 10
BATCH_SIZE = 512

# Download the MNIST dataset and prepare the dataset
# *******
# The MNIST database of handwritten digits, available from this page, 
# has a training set of 60,000 examples, and a test set of 10,000 examples.
# img size: 28x28
# *******
train_data = torchvision.datasets.MNIST('.', train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.MNIST('.', train=False, transform=torchvision.transforms.ToTensor(), download=True)
# Load the datasets
train_dataloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
# = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:1000]#/255. # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.targets[:1000]

# Build the network
class YuanNet(torch.nn.Module):
	def __init__(self):
		super(YuanNet, self).__init__()
		self.conv1 = torch.nn.Conv2d(
			in_channels=1, 
			out_channels=16, # n_filters 
			kernel_size=4, 
			stride=2) # input 28x28, output 13x13 -> BATCH_SZIE x 16 x 13 x 13

		self.conv2 = torch.nn.Conv2d(
			in_channels=16, 
			out_channels=32, # n_filters 
			kernel_size=3, 
			stride=2) # input 13x13, output 6x6, -> BATCH_SZIE x 32 x 6 x 6

		#self.avgpool = torch.nn.AvgPool2d(kernel_size=6) # BATCH_SZIE x 32 x 1 x 1, remember use squeeze after avgpool 
		self.output = torch.nn.Linear(32*6*6, 10) # BATCH_SZIE x 10

	def forward(self, x): # x shape is BATCH_SZIE x 1 x 28 x 28 -> BATCH_SZIE x 28 x28
		x = self.conv1(x)
		x = F.relu(x)
		x = self.conv2(x)
		x = F.relu(x)
		#x = self.avgpool(x) # BATCH_SZIE x 32 x 1 x 1, remember use squeeze after avgpool 
		#x = x.squeeze() 
		x = x.view(x.size(0), -1) # Flatten it, 
		# torch.view() returns a new tensor with the same data as the self tensor but of a different shape
		#print(x.size())
		x = self.output(x)

		return x # BATCH_SZIE x 10

# Training
net = YuanNet()
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.02)

for epoch in range(EPOCH):
	for x ,y in train_dataloader:
		#print('x, y: ', x ,y)
		output = net(x)
		loss = loss_func(output, y) # predict label and ground true label
		optimizer.zero_grad() # clear gradients for this training step
		loss.backward() # backpropagation, compute gradients
		optimizer.step() # apply gradients
		#predict = torch.nn.Softmax(output)
		#result = torch.max(predict)[1]

	test_output = net(test_x)
	#print(test_output)
	predict = torch.max(test_output,1)[1]
	#print(predict)
	accuracy = (predict == test_y).sum().item() / float(test_y.size(0))
	# torch.item() returns the value of this tensor as a standard Python number. This only works for tensors with one element.
	print('epoch:{} | loss:{:.4f} | accuracy:{:.4f}'.format(epoch+1, loss, accuracy))