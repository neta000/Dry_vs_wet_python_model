from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
 
# load and prepare the image
def load_image(filename):
	# load the image
	img = load_img(filename, target_size=(200,200))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 3 channels
	img = img.reshape(1, 200, 200, 3)
	# center pixel data
	img = img.astype('float32')
	print(img)
	return img
 
# load an image and predict the class
def run_example():
	# load the image
	img = load_image('s3.jpg')
	# load model
	model = load_model('final_model_84%.h5')
	# predict the class
	result = model.predict(img)
	print(result)
 
	
 
# entry point, run the example
run_example()