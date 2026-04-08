"""
minicv.io
=========
Image I/O and color-space conversion utilities.
Uses matplotlib.image exclusively for reading and writing.
"""

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from .utils import to_float, to_uint8


def read_image(path, as_gray=False):
    """
    Read an image from disk into a NumPy array.

    Supports PNG (float32 [0,1] from matplotlib) and JPEG/other formats
    (uint8 [0,255]). The output is always normalised to uint8 [0,255].

    Parameters
    ----------
    path : str
        Filesystem path to the image file.
    as_gray : bool, optional
        If True, convert the loaded image to grayscale. Default is False.

    Returns
    -------
    np.ndarray
        - Shape (H, W, 3)   — RGB image (as_gray=False).
        - Shape (H, W)      — Grayscale image (as_gray=True).
        dtype is uint8 with values in [0, 255].

    Raises
    ------
    TypeError
        If `path` is not a string, or `as_gray` is not a bool.
    FileNotFoundError
        If the file does not exist at the given path.
    ValueError
        If the loaded array has an unexpected number of dimensions.
    """
    if not isinstance(path, str):
        raise TypeError(f"path must be a str, got {type(path).__name__}.")
    if not isinstance(as_gray, bool):
        raise TypeError(f"as_gray must be a bool, got {type(as_gray).__name__}.")

    import os
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No image file found at '{path}'.")

    img = mpimg.imread(path)

    if img.ndim not in (2, 3):
        raise ValueError(f"Unexpected image dimensions: {img.ndim}.")

    # matplotlib returns float32 in [0,1] for PNG; uint8 for JPEG, etc.
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = to_uint8(img.astype(np.float64))
    else:
        img = img.astype(np.uint8)

    # Drop alpha channel if present
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    # Handle already-grayscale files (shape H×W or H×W×1)
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]

    if as_gray:
        if img.ndim == 3:
            img = rgb_to_gray(img)
        # already 2D — nothing to do

    return img


def save_image(path, img):
    """
    Save a NumPy image array to disk.

    Parameters
    ----------
    path : str
        Destination file path. The format is inferred from the extension
        (e.g. ".png", ".jpg").
    img : np.ndarray
        Image array. Shape (H, W) or (H, W, 3). Accepted dtypes: uint8 or
        float (values in [0, 1]).

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If `path` is not a str or `img` is not a np.ndarray.
    ValueError
        If `img` shape is not 2D or 3D, or has an unsupported number of channels.
    """
    if not isinstance(path, str):
        raise TypeError(f"path must be a str, got {type(path).__name__}.")
    if not isinstance(img, np.ndarray):
        raise TypeError(f"img must be a np.ndarray, got {type(img).__name__}.")
    if img.ndim not in (2, 3):
        raise ValueError(f"img must be 2D or 3D, got shape {img.shape}.")
    if img.ndim == 3 and img.shape[2] not in (3, 4):
        raise ValueError(f"img must have 3 or 4 channels, got {img.shape[2]}.")

    # Convert float images to uint8 for saving
    if img.dtype in (np.float32, np.float64):
        img = to_uint8(img)

    mpimg.imsave(path, img)


def rgb_to_gray(img):
    """
    Convert an RGB image to grayscale using the ITU-R BT.601 luma formula:
        Y = 0.299·R + 0.587·G + 0.114·B

    Parameters
    ----------
    img : np.ndarray
        RGB image of shape (H, W, 3). Values may be uint8 [0,255] or
        float [0,1].

    Returns
    -------
    np.ndarray
        Grayscale image of shape (H, W), same dtype as input.

    Raises
    ------
    TypeError
        If `img` is not a np.ndarray.
    ValueError
        If `img` is not 3D with exactly 3 channels.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(img).__name__}.")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(
            f"img must have shape (H, W, 3), got {img.shape}."
        )

    weights = np.array([0.299, 0.587, 0.114], dtype=np.float64)
    gray = (img.astype(np.float64) @ weights)

    # Preserve the original dtype
    if img.dtype == np.uint8:
        return np.clip(gray, 0, 255).astype(np.uint8)
    return gray.astype(img.dtype)


def gray_to_rgb(img):
    """
    Convert a grayscale image to a 3-channel RGB image by stacking the
    single channel three times.

    Parameters
    ----------
    img : np.ndarray
        Grayscale image of shape (H, W). Any numeric dtype.

    Returns
    -------
    np.ndarray
        RGB image of shape (H, W, 3), same dtype as input.

    Raises
    ------
    TypeError
        If `img` is not a np.ndarray.
    ValueError
        If `img` is not 2D.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(img).__name__}.")
    if img.ndim != 2:
        raise ValueError(f"img must be 2D (grayscale), got shape {img.shape}.")

    return np.stack([img, img, img], axis=-1)