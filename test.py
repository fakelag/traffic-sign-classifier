import model as mdl
import numpy as np
import pickle

from keras.utils.np_utils import to_categorical

# Load test data
with open("german-traffic-signs/test.p", "rb") as fp:
	test_data = pickle.load(fp)

x_test, y_test = test_data["features"], test_data["labels"]

# Validate sets
assert(x_test.shape[0] == y_test.shape[0])
assert(x_test.shape[1:] == (32, 32, 3))

# Format test data
x_test = np.array(list(map(mdl.image_normalize, x_test)))
x_test = x_test.reshape(x_test.shape[0], 32, 32, 1)
y_test = to_categorical(y_test, mdl.num_classes)

# Load the model
model = mdl.model_load()

# Evaluate
score = model.evaluate(x_test, y_test, verbose=0)
print("Test accuracy: ", score)
