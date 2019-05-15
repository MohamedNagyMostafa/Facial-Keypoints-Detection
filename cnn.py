import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.nn import Module

class Net(Module):

	def __init__(self):
		super(Net, self).__init__()

		input_depth = 1
		output_depth = 16
		kernel = 3
		padding = 1
		# 224 x 224 x 1
		self.conv1 		= nn.Conv2d(input_depth, 32, kernel_size=kernel)
		self.conv1_pull = nn.Conv2d(input_depth, 64, kernel_size=kernel)
		# 224 x 224 x 32
		# 112 x 112 x 32
		self.conv2 		= nn.Conv2d(32, 64, kernel_size=kernel)
		self.conv2_pull = nn.Conv2d(32, 128, kernel_size=kernel)
		# 112 x 112 x 64
		# 56 x 56 x 64
		self.conv3 		= nn.Conv2d(64,128, kernel_size=kernel)
		self.conv3_pull = nn.Conv2d(64,168, kernel_size=kernel)
		# 56 x 56 x 128
		# 28 x 28 x 128
		self.conv4 		= nn.Conv2d(128, 168, kernel_size=kernel)
		self.conv4_pull = nn.Conv2d(128, 256, kernel_size=kernel)
		# 28 x 28 x 168
		# 14 x 14 x 168
		self.conv5 		= nn.Conv2d(168, 256, kernel_size=kernel)
		# 10 x 10 x 256
		# 5 x 5 x 256

		self.dropout = nn.Dropout(p=0.15)
		self.pooling = nn.MaxPool2d(2,2)
		# Change 5 * 5
		self.cl1 = 5 * 5 * 256
		# Change 13*13
		self.cl2 = 12 * 12 * 256
		self.cl3 = 12 * 12 * 256
		self.cl4 = 12 * 12 * 256

		self.fc1 = nn.Linear(self.cl1 + self.cl2 + self.cl3 + self.cl4, 500)
		self.fc2 = nn.Linear(500, 230)
		self.fc3 = nn.Linear(230, 136)

	def forward(self, x):

		# Conv ->1-2-3-4-5
		layer1 = self.pooling(func.relu(self.conv1(x)))
		layer2 = self.pooling(func.relu(self.conv2(layer1)))
		layer3 = self.pooling(func.relu(self.conv3(layer2)))
		layer4 = self.pooling(func.relu(self.conv4(layer3)))
		layer5 = self.pooling(func.relu(self.conv5(layer4)))
		# Conv ->1-3-4-5
		layer1_pull = self.pooling(func.relu(self.conv1_pull(x)))
		layer1_3pull = self.pooling(func.relu(self.conv3(layer1_pull)))
		layer1_4pull = self.pooling(func.relu(self.conv4(layer1_3pull)))
		layer1_5pull = self.pooling(func.relu(self.conv5(layer1_4pull)))
		# Conv ->2-4-5
		layer2_pull = self.pooling(func.relu(self.conv2_pull(layer1)))
		layer2_4pull = self.pooling(func.relu(self.conv4(layer2_pull)))
		layer2_5pull = self.pooling(func.relu(self.conv5(layer2_4pull)))
		# Cpnv ->3-5
		layer3_pull  = self.pooling(func.relu(self.conv3_pull(layer2)))
		layer3_5pull = self.pooling(func.relu(self.conv5(layer3_pull)))
		
		f1 = layer5.view(-1, self.cl1)
		f2 = layer1_5pull.view(-1, self.cl2)
		f3 = layer2_5pull.view(-1, self.cl3)
		f4 = layer3_5pull.view(-1, self.cl4)

		f = torch.cat((f1,f2,f3,f4), 1)

		return self.fc3(self.dropout(func.relu(self.fc2(self.dropout(func.relu(self.fc1(f)))))))
