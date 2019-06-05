import numpy as np

with open("data/train-images-idx3-ubyte", "rb", 0) as f:
	meta = f.read(128)
	image = np.array([])
	extra = f.read(eval('784*2'))
	for i in range(784):
		image = np.append(image, int.from_bytes(f.read(1), 'big'))
		

# Digit showcase block
import matplotlib.pyplot as plt
pixels = np.reshape(image, (28,28))
plt.imshow(pixels, cmap='gray')
plt.show()