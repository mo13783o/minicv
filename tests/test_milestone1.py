import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np

print("========== MINICV TEST ==========\n")

# -------------------------------
# 1) Import Test
# -------------------------------
try:
    from minicv.utils import pad_image, normalize_image, clip_image
    from minicv.filters import convolve2d, mean_filter, gaussian_filter
    from minicv.transforms import resize, rotate, translate
    from minicv.features import compute_histogram
    from minicv.drawing import draw_point
    print("✅ Import Test: PASS\n")
except Exception as e:
    print("❌ Import Test: FAIL", e)
    exit()

# -------------------------------
# 2) Utils Test
# -------------------------------
try:
    img = np.array([[1, 2], [3, 4]])

    padded = pad_image(img, 1)
    assert padded.shape == (4, 4)

    norm = normalize_image(img)
    assert norm.min() >= 0 and norm.max() <= 1

    clipped = clip_image(img, 0, 3)
    assert clipped.max() <= 3

    print("✅ Utils Test: PASS\n")
except Exception as e:
    print("❌ Utils Test: FAIL", e)

# -------------------------------
# 3) Convolution Test
# -------------------------------
try:
    img = np.array([[1,2,3],
                    [4,5,6],
                    [7,8,9]])

    kernel = np.ones((3,3)) / 9
    out = convolve2d(img, kernel)

    assert out.shape == img.shape

    print("✅ Convolution Test: PASS\n")
except Exception as e:
    print("❌ Convolution Test: FAIL", e)

# -------------------------------
# 4) Filters Test
# -------------------------------
try:
    img = np.random.rand(5,5)

    mean = mean_filter(img, 3)
    gauss = gaussian_filter(img, 3, 1.0)

    assert mean.shape == img.shape
    assert gauss.shape == img.shape

    print("✅ Filters Test: PASS\n")
except Exception as e:
    print("❌ Filters Test: FAIL", e)

# -------------------------------
# 5) Transform Test
# -------------------------------
try:
    img = np.random.rand(5,5)

    r = resize(img, (10,10))
    rot = rotate(img, 45)
    t = translate(img, 1, 1)

    assert r.shape == (10,10)

    print("✅ Transform Test: PASS\n")
except Exception as e:
    print("❌ Transform Test: FAIL", e)

# -------------------------------
# 6) Feature Test
# -------------------------------
try:
    img = (np.random.rand(10,10)*255).astype(np.uint8)

    hist = compute_histogram(img)

    assert hist.sum() == img.size

    print("✅ Feature Test: PASS\n")
except Exception as e:
    print("❌ Feature Test: FAIL", e)

# -------------------------------
# 7) Drawing Test
# -------------------------------
try:
    img = np.zeros((10,10))

    out = draw_point(img.copy(), 5, 5, 1)

    assert out[5,5] == 1

    print("✅ Drawing Test: PASS\n")
except Exception as e:
    print("❌ Drawing Test: FAIL", e)

# -------------------------------
print("========== TEST FINISHED ==========")