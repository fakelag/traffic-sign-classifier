import numpy as np
import matplotlib.pyplot as plt
import model as mdl
import keras
import random
import requests
import pickle

from PIL import Image
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

# Load training data
with open("german-traffic-signs/train.p", "rb") as fp:
	train_data = pickle.load(fp)

with open("german-traffic-signs/valid.p", "rb") as fp:
	val_data = pickle.load(fp)

x_train, y_train = train_data["features"], train_data["labels"]
x_val, y_val = val_data["features"], val_data["labels"]

# Validate sets
assert(x_train.shape[0] == y_train.shape[0])
assert(x_val.shape[0] == y_val.shape[0])

# Format training data
x_train = np.array(list(map(mdl.image_normalize, x_train)))
x_val = np.array(list(map(mdl.image_normalize, x_val)))

x_train = x_train.reshape(x_train.shape[0], 32, 32, 1)
x_val = x_val.reshape(x_val.shape[0], 32, 32, 1)

y_train = to_categorical(y_train, mdl.num_classes)
y_val = to_categorical(y_val, mdl.num_classes)

# Generate variance in the dataset
image_gen = ImageDataGenerator(
	width_shift_range=0.1,
	height_shift_range=0.1,
	zoom_range=0.2,
	shear_range=0.1,
	rotation_range=10
)

image_gen.fit(x_train)
batches = image_gen.flow(x_train, y_train, batch_size=15)

x_batch, y_batch = next(batches)

# Create, Train & Save the model
model = mdl.model_create()
history = mdl.model_train_with_generator(model, image_gen, x_train, y_train, x_val, y_val) # mdl.model_train(model, x_train, y_train, x_val, y_val)

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.legend(["loss", "val_loss"])
plt.title("Loss")
plt.xlabel("epoch")

plt.show()
