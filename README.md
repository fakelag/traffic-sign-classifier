# Traffic Sign Classifier

Classifies traffic sign images into corresponding categories using deep convolutional neural networks

## CNN Layout
| Layer (type) | Output Shape | Param #  |
| ------------- | ------------- | ------------- |
| conv2d_1 (Conv2D) | (None, 28, 28, 60) | 1560 |
| conv2d_2 (Conv2D) | (None, 24, 24, 60) | 90060 |
| max_pooling2d_1 (MaxPooling2) | (None, 12, 12, 60) | 0 |
| conv2d_3 (Conv2D) | (None, 10, 10, 30) | 16230 |
| conv2d_4 (Conv2D) | (None, 8, 8, 30) | 8130 |
| max_pooling2d_2 (MaxPooling2) | (None, 4, 4, 30) | 0 |
| flatten_1 (Flatten) | (None, 480) | 0 |
| dense_1 (Dense) | (None, 500) | 240500 |
| dropout_1 (Dropout) | (None, 500) | 0 |
| dense_2 (Dense) | (None, 43) | 21543 |

## Training data
https://bitbucket.org/jadslim/german-traffic-signs
