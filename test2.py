import warnings
warnings.filterwarnings("ignore")

from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

# Load our trained model
use_case = 'vaguelettes'
model = load_model('./models/trained_model')

# Apply it to all pictures present in the piture test folder
img_folder = './pictures/test/'
res_file = 'prediction.txt'
img_list = os.listdir(img_folder)
with open(res_file, 'a') as f:
	f.write('\t'.join(['image_name'] + os.listdir('./pictures/' + use_case + '/train/') + ['\n']))
	for img_name in img_list:
		print(img_name)
		
		img = image.load_img(img_folder + img_name, target_size=(224, 224))
		
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
		
		pred = model.predict(x)
		print(pred)
		
		f.write('\t'.join([img_name, str(pred), '\n']))

