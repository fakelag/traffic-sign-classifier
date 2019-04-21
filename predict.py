import numpy as np
import matplotlib.pyplot as plt
import model as mdl
import requests
import cv2

from PIL import Image

# Download an image from the interwebs
url = "https://previews.123rf.com/images/pejo/pejo0907/pejo090700003/5155701-german-traffic-sign-no-205-give-way.jpg"
r = requests.get(url, stream=True)
img = Image.open(r.raw)
img_original = img

# Preview it
plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.show()

# Convert it
img = mdl.image_process(img)

# Load the model
model = mdl.model_create()
model = mdl.model_load(model)

# Perform prediction
print("predicted sign: " + str(model.predict_classes(img)))
