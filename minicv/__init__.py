"""
minicv
======
A small OpenCV-like image processing library built from scratch with NumPy.

Quick start
-----------
>>> from minicv import read_image, gaussian_filter, resize, save_image
>>> img = read_image("photo.png")
>>> blurred = gaussian_filter(img, kernel_size=7, sigma=2.0)
>>> small = resize(blurred, (128, 128))
>>> save_image("output.png", small)
"""

# ── I/O ─────────────────────────────────────────────────────────────────────
from .io import (
    read_image,
    save_image,
    rgb_to_gray,
    gray_to_rgb,
)

# ── Utilities ────────────────────────────────────────────────────────────────
from .utils import (
    normalize_image,
    clip_image,
    pad_image,
    to_float,
    to_uint8,
)

# ── Filters ──────────────────────────────────────────────────────────────────
from .filters import (
    convolve2d,
    apply_filter,
    mean_filter,
    gaussian_filter,
    median_filter,
    sobel_gradients,
)

# ── Transforms ───────────────────────────────────────────────────────────────
from .transforms import (
    resize,
    rotate,
    translate,
)

# ── Features ─────────────────────────────────────────────────────────────────
from .features import (
    compute_histogram,
    histogram_equalization,
    sobel_features,
)

# ── Drawing ──────────────────────────────────────────────────────────────────
from .drawing import (
    draw_point,
    draw_line,
    draw_rectangle,
    draw_polygon,
)

__all__ = [
    # io
    "read_image", "save_image", "rgb_to_gray", "gray_to_rgb",
    # utils
    "normalize_image", "clip_image", "pad_image", "to_float", "to_uint8",
    # filters
    "convolve2d", "apply_filter", "mean_filter", "gaussian_filter",
    "median_filter", "sobel_gradients",
    # transforms
    "resize", "rotate", "translate",
    # features
    "compute_histogram", "histogram_equalization", "sobel_features",
    # drawing
    "draw_point", "draw_line", "draw_rectangle", "draw_polygon",
]

__version__ = "0.1.0"
__author__  = "minicv contributors"