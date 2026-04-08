import numpy as np
from minicv.filters import convolve2d

img = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

kernel = np.ones((3,3)) / 9

output = convolve2d(img, kernel)

print(output)