from os import listdir
from numpy import asarray
from numpy import save
import numpy
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
# define location of dataset
folder = 'train/'
photos, labels = list(), list()
# enumerate files in the directory
for file in listdir(folder):
	# determine class
	output = 0
	if file.startswith('w'):
		output = 1
	# load image
	photo = load_img(folder + file, target_size=(200, 200))
	# convert to numpy array
	photo = img_to_array(photo,dtype="uint8")
	# store
	photos.append(photo)
	labels.append(output)
# convert to a numpy arrays
photos = asarray(photos)
labels = asarray(labels)
numpy.uint8(photos)
numpy.uint8(labels)
print(photos.dtype)
print(labels.dtype)

print(photos.shape, labels.shape)
# save the reshaped photos
save('d_vs_w.npy', photos)
save('d_vs_w_labels.npy', labels)