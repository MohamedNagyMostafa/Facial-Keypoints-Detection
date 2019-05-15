import numpy as np
from utils import *

#read an image
image = readImage('obamas.jpg',3)

#display image
show(image)

#detect faces in image
faces = cascadeFaces(image)

#display image with cascaded faces
image_detected_Faces = np.copy(image)
image_faces = showFacesCascade(image, faces)

#using the trained model to get facial for each face
output = feedFacialModel(image_faces)

#display result
show_ky(output)
