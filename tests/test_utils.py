import numpy as np
from minicv.utils import normalize_image

def test_normalize():
    img = np.array([0, 5, 10])
    out = normalize_image(img, mode="minmax")

    assert out.min() == 0
    assert out.max() == 1