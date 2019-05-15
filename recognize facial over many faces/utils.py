import cv2 
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.abspath(os.path.join('..')))
from cnn import Net
from transform import *
from torchvision import transforms

def readImage(name, channels):
	image = cv2.imread('images/' + name)
	if channels == 1 :
		return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def show(image):
	if len(image.shape) > 2 and image.shape[2] == 3:
		plt.imshow(image)
	else:
		plt.imshow(image, cmap='gray')
	plt.show()

def cascadeFaces(image):
	face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')
	return face_cascade.detectMultiScale(image,1.5,2)

def showFacesCascade(image, faces):
	faces_image = []
	
	image_cp = np.copy(image)
	for (x,y,w,h) in faces:
		out_w = int(w/5)
		out_h = int(h/5)
		faces_image.append(image_cp[y-out_h:y+h+out_h,x-out_w:x+w+out_w])
		cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 3)
	show(image)
	return faces_image

def feedFacialModel(images):

	images_copied = np.copy(images)
	output = []

	input = preprocessing(images)
	
	out = model(input)

	output = postProcessing(out, input)

	return output

def model(input):
	net = Net()
	net = loadModel(net)

	return net(input)


def loadModel(net):
	model_dir = '../SavedModels/'
	model_name = 'pull_model_saved'
	check_point = torch.load(model_dir+model_name)
	net.load_state_dict(check_point)
	return net


def postProcessing(out, images):
	out = out.view(len(images), -1, 2)
	out = out.data * 50.0 + 100
	images = np.transpose(images,(0,2,3,1)).numpy().squeeze(3)

	samples = []
	for i in range(len(images)):
		image = images[i]
		keypoints = out[i]

		samples.append({'image':image, 'keypoints':keypoints})

	return samples


def preprocessing(images):
	for i,image in enumerate(images):
		image = cv2.resize(image, (223,223))
		image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		image = image/255
		images[i] = image

	input = torch.tensor(images).unsqueeze(1)
	print(input.shape)
	input = input.type(torch.FloatTensor)
	return input

def show_ky(outputs):
	fig = plt.figure(figsize=(9,9))
	images_count = len(outputs)
	row, col = int(images_count/2), int(images_count/2) if images_count % 2 == 0 else int(images_count/2) + 1
	if images_count == 2:
		col = 2
	for i, sample in enumerate(outputs):
		image = sample['image']
		keypoints = sample['keypoints']

		fig.add_subplot(row, col, (i % col) + 1)
		plt.imshow(image, cmap='gray')
		plt.scatter(keypoints[:,0], keypoints[:,1], s=20, marker='.', c='m')

	plt.show()