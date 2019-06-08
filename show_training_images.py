from numpy import *
m = 1
bytes_to_read  = m * 784
with open("data/train-images-idx3-ubyte", "rb") as f:
    meta = f.read(16)
    # raw_images = f.read(47040000)
    raw_images = f.read(bytes_to_read)
x = reshape(array([raw_images[i] for i in range(bytes_to_read)]), (m,784))
with open("data/train-labels-idx1-ubyte", "rb") as f:
    meta = f.read(8)
    raw_labels = f.read(m)
y = reshape(array([raw_labels[i] for i in range(m)]), (m,1))

import matplotlib.pyplot as plt
pixels = reshape(x, (28,28))
plt.title('Shown image is for ' + str(y))
plt.imshow(pixels, cmap='gray')
plt.show()