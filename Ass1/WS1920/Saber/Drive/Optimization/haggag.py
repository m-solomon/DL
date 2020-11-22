import numpy as np


a = np.ones((10,10,3))

b = np.sum(a, axis=(1,2))

print(b, b.shape)