from numpy import *

# Parsing the training data and their labels
with open("data/train-images-idx3-ubyte", "rb") as f:
    meta = f.read(16)
    raw_images = f.read(47040000)
training_images = reshape(array([raw_images[i] for i in range(47040000)]), (60000,784))

with open("data/train-labels-idx1-ubyte", "rb") as f:
    meta = f.read(8)
    raw_labels = f.read(60000)
training_labels = reshape(array([raw_labels[i] for i in range(60000)]), (60000,1))
# print(training_labels[-1])

# import matplotlib.pyplot as plt
# pixels = reshape(training_images[-1], (28,28))
# plt.imshow(pixels, cmap='gray')
# plt.show()
# the last image was 8