from keras.optimizers import Adam
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D

import numpy as np
import pandas as pd
import cv2

# Class definitions
class_defs = pd.read_csv("german-traffic-signs/signnames.csv")
num_classes = 43

def model_save(mdl):
	""" Saves the model to disk """
	mdl.save("model.h5")

def model_load():
	""" Loads the model from disk """
	return load_model("model.h5")

def model_create():
	""" Create a convolutional neural network model (CNN) """
	mdl = Sequential()

	# Add the first convolutional layer
	mdl.add(Conv2D(60, (5, 5), input_shape=(32, 32, 1), activation='relu'))
	mdl.add(Conv2D(60, (5, 5), activation='relu'))
	mdl.add(MaxPooling2D(pool_size=(2, 2)))

	# Add the second convolutional layer
	mdl.add(Conv2D(30, (3, 3), activation='relu'))
	mdl.add(Conv2D(30, (3, 3), activation='relu'))
	mdl.add(MaxPooling2D(pool_size=(2, 2)))

	mdl.add(Flatten())
	mdl.add(Dense(500, activation='relu'))

	# Add a dropout to adjust training
	mdl.add(Dropout(0.5))

	# Add a softmax output
	mdl.add(Dense(num_classes, activation='softmax'))

	# Compile the model
	mdl.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
	return mdl

def model_train(model, x_train, y_train, x_val, y_val):
	""" Train the model with a single set of data """
	h = model.fit(x_train, y_train, batch_size=400, epochs=10, validation_data=(x_val, y_val), verbose=1, shuffle=1)
	model_save(model)
	return h

def model_train_with_generator(model, datagen, x_train, y_train, x_val, y_val):
	""" Train the model with a single dataset + an image generator """
	h = model.fit_generator(datagen.flow(x_train, y_train, batch_size=50), steps_per_epoch=2000, epochs=10, validation_data=(x_val, y_val), shuffle=1)
	model_save(model)
	return h

def image_normalize(img):
	""" Grayscale & Normalize the image, return a single depth channel image (0.0-1.0) """
	image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	image = cv2.equalizeHist(image)
	return image / 255

def image_process(img):
	""" Normalize & Reshape the image to conform to the neural network input shape """
	img = np.asarray(img)
	img = cv2.resize(img, (32, 32))
	img = image_normalize(img)
	return img.reshape(1, 32, 32, 1)
	