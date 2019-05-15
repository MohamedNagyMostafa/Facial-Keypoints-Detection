from facial_dataset import FacialDataset
from transform import Rescale, ToTensor, Normalize, RandomCrop
from torchvision import transforms
import matplotlib.pyplot as plt
from cnn import Net
import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader
import torch.optim as optim
from transform import *

def getImage(image):
	return np.transpose(image,(1,2,0)).numpy().squeeze(2)

def reprocessing(image, key_pts, key_pts_2):
	return np.transpose(image,(0,2,3,1)).numpy().squeeze(3), key_pts.data * 50.0 + 100, key_pts_2.data * 50.0 + 100

def loadModel(model, optimizer):
	model_dir = 'SavedModels/'
	model_name = 'keypoints_model_6_Loss1.pt'
	check_point = torch.load(model_dir+model_name)
	model.load_state_dict(check_point['model_state'])
	optimizer.load_state_dict(check_point['optimizer_state'])
	return model, optimizer

def saveModel(model, optimizer):
	model_dir = 'SavedModels/'
	model_name = 'keypoints_model_6_Loss1.pt'
	torch.save({'model_state':model.state_dict(), 'optimizer_state':optimizer.state_dict()}, model_dir+model_name)

def display(image, key_points):
	plt.imshow(image, cmap='gray')
	plt.scatter(key_points[:,0], key_points[:,1])
	plt.show()

def model_output(model, testLoader, batch_size):
	for idx, sample in enumerate(testLoader):
		images = sample['image']
		key_pts = sample['keypoints']

		images = images.type(torch.FloatTensor)
		out = model(images)
		out = out.view(1, -1, 2)
		if idx == 0:
			return images, out, key_pts

def train(model, optimizer, epoch, critirion, train_loader, testLoader, batch_size):
	if torch.cuda.is_available():
		model = model.cuda()
	least_cost = 0.0015641744712718412
	for epoch in range(epoch):
		train_loss = 0
		test_loss = 0
		model.train()
		for sample in trainLoader:
			images = sample['image']
			target = sample['keypoints']
			#flatting
			target = target.view(target.size(0), -1)

			target = target.type(torch.FloatTensor)
			images = images.type(torch.FloatTensor)
			if torch.cuda.is_available():
				images = images.cuda()
				target = target.cuda()

			out = model(images)
			cost = critirion(out, target)

			optimizer.zero_grad()

			cost.backward()
			optimizer.step()

			train_loss += cost.item()

		else:
			model.eval()

			for sample in testLoader:
				images = sample['image']
				target = sample['keypoints']
				#flatting
				target = target.view(target.size(0), -1)

				target = target.type(torch.FloatTensor)
				images = images.type(torch.FloatTensor)
				if torch.cuda.is_available():
					images = images.cuda()
					target = target.cuda()

				out = model(images)
				cost = critirion(out, target)

				test_loss += cost.item()
		train_loss = train_loss/len(trainLoader)
		test_loss = test_loss/len(testLoader)

		if test_loss < least_cost:
			
			least_cost = test_loss
			saveModel(model, optimizer)

		print('epoch: {} Tloss: {} test LOss: {}'.format(epoch+1, train_loss, test_loss))



def test(model, testLoader, batch_size, im_num, savedModel=True):

	image, out, key_pts = model_output(model, testLoader, batch_size)
	image, out, key_pts = reprocessing(image, out, key_pts )
	for idx in range(im_num):
		display(image[idx], out.data[idx])
		display(image[idx], key_pts.data[idx])		

def feature_visualization(weights, image, depth):
	fig = plt.figure(figsize=(20,8))
	el = depth
	depth = np.sqrt(depth)
	row = np.round(depth).astype(int) if depth - np.round(depth) == 0 else (np.round(depth) + 1).astype(int)
	column = row * 2

	for i in range(0,el):
		
		fig.add_subplot(row,column,(i*2)+1)
		plt.imshow(weights[i][0], cmap='gray')
		fig.add_subplot(row,column,(i*2)+2)
		plt.imshow(cv2.filter2D(image, -1, weights[i][0]), cmap='gray')
			

	plt.show()
#=================================================================
csv_dir_train = 'data/training_frames_keypoints.csv'
root_dir_train = 'data/training/'
csv_dir_test = 'data/test_frames_keypoints.csv'
root_dir_test = 'data/test/'

batch_size = 8

transform = transforms.Compose([
	RandomRotation(15),
	Rescale(224),
	RandomCrop(223),
	Normalize(),
	ToTensor()
])

trainDataset = FacialDataset(csv_dir_train, root_dir_train, transform)
testDataset = FacialDataset(csv_dir_test, root_dir_test, transform)
'''

sample = trainDataset[0]
angle = 22.5

sample_x = transform(sample)
display(sample_x['image'], sample_x['keypoints'])

print(sample['keypoints'][0][1])
'''

trainLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True, num_workers=0)
testLoader = DataLoader(testDataset, batch_size=1, shuffle=True, num_workers=0)

model = Net()
critirion = torch.nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)
print(model)
epoch = 1000

model.cuda()
model, optimizer = loadModel(model, optimizer)

#train(epoch=epoch, train_loader=trainLoader, optimizer= optimizer, critirion=critirion, model=model, testLoader= testLoader, batch_size= batch_size)
torch.save(model.state_dict(), 'SavedModels/pull_model_saved')
#test(model=model, testLoader=testLoader, batch_size=batch_size, im_num=12)

#model = loadModel(model)

#weights = model.conv2.weight.data.numpy()
#feature_visualization(weights=weights, image=getImage(iter(testLoader).next()['image'][0]), depth =32)

### Test

def read_image(dir):
	transform = transforms.Compose([
	Rescale(224),
	RandomCrop(223),
	Normalize(),
	ToTensor()
	])

	image = cv2.imread(dir)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	sample = {'image': image, 'keypoints': None}
	transformed_image = transform(sample)

	images = transformed_image['image']
	print(images.shape)
	images = images.type(torch.FloatTensor)
	images = images.unsqueeze(0)
	out = model(images)
	out = out.view(-1, 2)
	out = out.data * 50.0 + 100
	images = np.transpose(images,(0,2,3,1)).numpy().squeeze(3)
	display(images.squeeze(0), out)
read_image('mohamed.jpg')
