import numpy as np
import cv2
import torch
import imutils

class Normalize(object):

	def __call__(self, sample):
		image, key_pts = sample['image'], sample['keypoints']

		image_copy = np.copy(image)
		key_pts_copy = np.copy(key_pts)

		image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
		image_copy = image_copy/255

		#re-range [-1 -> 1]
		if not key_pts is None:
			key_pts_copy = (key_pts_copy - 100.0)/50.0
		else:
			return {'image': image_copy, 'keypoints': None}
		sample = {'image': image_copy, 'keypoints': key_pts_copy}

		return sample

class RandomRotation(object):
	def __init__(self, angle):
		self.angle = angle

	def __call__(self, sample):
		angle = np.random.randint(0,self.angle)
		image, key_pts = sample['image'], sample['keypoints']
		#rotate the image
		image = imutils.rotate(image, angle)
		#rotate keypoints
		y_c,x_c = image.shape[0]/2, image.shape[1]/2

		#coordinate transformation
		angle = np.deg2rad(angle)

		key_pts[:,0], key_pts[:,1] = key_pts[:,0] - x_c, y_c - key_pts[:, 1]
		
		key_pts = np.array([[x * np.sin(angle) + y * np.cos(angle), x * np.cos(angle) - y * np.sin(angle)] for x,y in zip(key_pts[:,0], key_pts[:,1])])
		#coordinate transformation to original coordinates

		key_pts[:,0], key_pts[:,1] = key_pts[:,1] + x_c, y_c - key_pts[:, 0]
		return {'image': image, 'keypoints': key_pts}


class Rescale(object):

	def __init__(self, output_size):
		assert isinstance(output_size,(int, tuple))
		self.output_size = output_size

	def __call__(self, sample):
		image, key_pts = sample['image'], sample['keypoints']

		h, w = image.shape[:2]

		if isinstance(self.output_size, int):
			if h > w:
				new_w, new_h = self.output_size, self.output_size * h/w
			else:
				new_w, new_h = self.output_size * w/h, self.output_size
		else:
			new_w, new_h = self.output_size

		new_w, new_h = int(new_w), int(new_h)

		img = cv2.resize(image, (new_w, new_h))
		if not key_pts is None:
			key_pts = key_pts * [new_w/w, new_h/h]
		else:
			return {'image': img, 'keypoints': None}

		return {'image': img, 'keypoints': key_pts}

class RandomCrop(object):

	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple))
		if isinstance(output_size,(int)):
			self.output_size = (output_size, output_size)
		else:
			assert len(output_size) == 2
			self.output_size = output_size

	def __call__(self, sample):
		image, key_pts = sample['image'], sample['keypoints']

		h,w = image.shape[:2]
		new_h, new_w = self.output_size
		#Ensure about size exceeding
		top = np.random.randint(0, h - new_h)
		left = np.random.randint(0, w - new_w)
		image = image[top: top + new_h, left: left+ new_w]
		if not key_pts is None:
			key_pts = key_pts - [left, top]

		return {'image': image, 'keypoints': key_pts}

class ToTensor(object):

	def __call__(self, sample):
		image, key_pts = sample['image'], sample['keypoints']

		if(len(image.shape) == 2):
			image = image.reshape(image.shape[0], image.shape[1], 1)

		image = image.transpose((2,0,1))
		if not key_pts is None:
			return {'image':torch.from_numpy(image),
			'keypoints':torch.from_numpy(key_pts)}
		else:
			return {'image':torch.from_numpy(image),
			'keypoints':None}
