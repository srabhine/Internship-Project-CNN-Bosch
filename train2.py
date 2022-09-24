import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from keras.models import load_model

from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D

# SETTINGS VARIABLES
use_case = 'vaguelettes'

train_dir = './pictures/' + use_case + '/train/'
validatoin_dir = './pictures/' + use_case + '/validation/'

pic_class = os.listdir(train_dir)
nbr_class = len(pic_class)

nbr_epoch = 6
nbr_steps_per_epoch = 5
nbr_steps_validation = 5

train_base_layers = False

# MODEL INITIALIZING
# Load Base model for image featuring
base_model_path = './models/VGG16'
if os.path.exists(base_model_path):
            base_model = load_model(base_model_path)
else:
            from keras.applications.vgg16 import VGG16
            base_model = VGG16(weights='imagenet', include_top=False)
            base_model.save(base_model_path)

# Add finnal layers for classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(10, activation='relu')(x)
predictions = Dense(nbr_class, activation='softmax')(x)

# Set the Base model layers 
for layer in base_model.layers:
            layer.trainable = train_base_layers

model = Model(inputs=base_model.input, output=predictions)
model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=["accuracy"])


# PICTURE PREPROCESSING
# Generator for inputs enancement
datagen = image.ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=5,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=False)


train_gen = datagen.flow_from_directory(train_dir, target_size=(224,224), batch_size=32, class_mode='categorical')
validation_gen = datagen.flow_from_directory(validatoin_dir, target_size=(224,224), batch_size=32, class_mode='categorical')


# MODEL TRAINING
# Run the training process
H = model.fit_generator(train_gen, steps_per_epoch = nbr_steps_per_epoch, epochs=nbr_epoch, 
                                                                        validation_data=validation_gen, validation_steps=nbr_steps_validation)

# Save the model
model.save('./models/trained_model')

pd.DataFrame(H.history).to_csv('./models/trained_model_history.csv')

# Diplay the Training results
import matplotlib.pyplot as plt

N = nbr_epoch
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower right")

plt.show()
