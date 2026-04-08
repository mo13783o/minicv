
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import matplotlib.pyplot as plt

from minicv.io import read_image
from minicv.filters import gaussian_filter
from minicv.features import histogram_equalization, sobel_features

# -------------------------------
# 1) Load Image
# -------------------------------
current_dir = os.path.dirname(__file__)
image_path = os.path.join(current_dir, "..", "images", "test.jpg")

img = read_image(image_path, as_gray=True)

# -------------------------------
# 2) Apply Filters
# -------------------------------
blur = gaussian_filter(img, kernel_size=5, sigma=1.0)

# -------------------------------
# 3) Histogram Equalization
# -------------------------------
equalized = histogram_equalization(img)

# -------------------------------
# 4) Sobel Features
# -------------------------------
sobel = sobel_features(img)
edges = sobel["magnitude"]

# -------------------------------
# 5) Display Results
# -------------------------------
plt.figure(figsize=(12, 8))

# Original
plt.subplot(2, 2, 1)
plt.title("Original")
plt.imshow(img, cmap='gray')
plt.axis('off')

# Gaussian Blur
plt.subplot(2, 2, 2)
plt.title("Gaussian Blur")
plt.imshow(blur, cmap='gray')
plt.axis('off')

# Histogram Equalization
plt.subplot(2, 2, 3)
plt.title("Histogram Equalization")
plt.imshow(equalized, cmap='gray')
plt.axis('off')

# Sobel Edges
plt.subplot(2, 2, 4)
plt.title("Sobel Edges")
plt.imshow(edges, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()