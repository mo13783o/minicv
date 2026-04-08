"""
minicv.utils
============
Foundation utilities for image normalization, clipping, and padding.
All other modules depend on this module.
"""

import numpy as np


def normalize_image(img, mode="minmax", out_range=(0, 1)):
    """
    Normalize image pixel values to a specified range.

    Parameters
    ----------
    img : np.ndarray
        Input image array (2D grayscale or 3D RGB). dtype will be cast to float64.
    mode : str, optional
        Normalization mode. Options:
            - "minmax"  : scales so min→out_range[0], max→out_range[1]
            - "zscore"  : zero-mean, unit-variance (ignores out_range)
            - "fixed"   : clips then scales from [0,255] → out_range
        Default is "minmax".
    out_range : tuple of (float, float), optional
        Target (min, max) for "minmax" and "fixed" modes. Default is (0, 1).

    Returns
    -------
    np.ndarray
        Normalized image as float64 array with same shape as input.

    Raises
    ------
    TypeError
        If `img` is not a numpy ndarray, or `out_range` is not a tuple/list.
    ValueError
        If `img` is not 2D or 3D, `mode` is unknown, or out_range[0] >= out_range[1].
    """
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(img).__name__}.")
    if img.ndim not in (2, 3):
        raise ValueError(f"Image must be 2D or 3D, got shape {img.shape}.")
    if not isinstance(out_range, (tuple, list)) or len(out_range) != 2:
        raise TypeError("out_range must be a tuple or list of two numbers.")
    lo, hi = float(out_range[0]), float(out_range[1])
    if lo >= hi:
        raise ValueError(f"out_range[0] must be < out_range[1], got {out_range}.")

    img = img.astype(np.float64)

    if mode == "minmax":
        mn, mx = img.min(), img.max()
        if mx == mn:
            return np.full_like(img, lo)
        return (img - mn) / (mx - mn) * (hi - lo) + lo

    elif mode == "zscore":
        mean, std = img.mean(), img.std()
        if std == 0:
            return np.zeros_like(img)
        return (img - mean) / std

    elif mode == "fixed":
        img = np.clip(img, 0, 255)
        return img / 255.0 * (hi - lo) + lo

    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose from 'minmax', 'zscore', 'fixed'.")


def clip_image(img, min_val=0, max_val=255):
    """
    Clip image pixel values to [min_val, max_val].

    Parameters
    ----------
    img : np.ndarray
        Input image array (any shape, any numeric dtype).
    min_val : float, optional
        Lower bound. Default is 0.
    max_val : float, optional
        Upper bound. Default is 255.

    Returns
    -------
    np.ndarray
        Clipped array with same shape and dtype as input.

    Raises
    ------
    TypeError
        If `img` is not a numpy ndarray.
    ValueError
        If min_val >= max_val.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(img).__name__}.")
    if min_val >= max_val:
        raise ValueError(f"min_val ({min_val}) must be less than max_val ({max_val}).")
    return np.clip(img, min_val, max_val)


def pad_image(img, pad_width, mode="constant", constant_value=0):
    """
    Pad a 2D or 3D image array along the spatial (H, W) dimensions.

    Parameters
    ----------
    img : np.ndarray
        Input image. Shape: (H, W) or (H, W, C).
    pad_width : int or tuple
        Number of pixels to pad on each spatial side.
        - int  → same padding applied to top/bottom/left/right.
        - tuple of two ints (ph, pw) → ph rows, pw cols.
        - tuple of four ints (top, bottom, left, right) → asymmetric.
    mode : str, optional
        Padding strategy. Options:
            - "constant" : fills with `constant_value` (default 0).
            - "reflect"  : reflects values at the border (no border duplication).
            - "edge"     : replicates edge pixels.
        Default is "constant".
    constant_value : float, optional
        Fill value used when mode="constant". Default is 0.

    Returns
    -------
    np.ndarray
        Padded image with same dtype as input.

    Raises
    ------
    TypeError
        If `img` is not a numpy ndarray.
    ValueError
        If `img` is not 2D or 3D, `pad_width` format is invalid, or `mode` is unknown.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(img).__name__}.")
    if img.ndim not in (2, 3):
        raise ValueError(f"Image must be 2D or 3D, got shape {img.shape}.")

    # Normalise pad_width → (top, bottom, left, right)
    if isinstance(pad_width, int):
        top = bottom = left = right = pad_width
    elif isinstance(pad_width, (tuple, list)):
        if len(pad_width) == 2:
            top = bottom = int(pad_width[0])
            left = right = int(pad_width[1])
        elif len(pad_width) == 4:
            top, bottom, left, right = (int(v) for v in pad_width)
        else:
            raise ValueError("pad_width tuple must have 2 or 4 elements.")
    else:
        raise TypeError("pad_width must be an int or a tuple of 2 or 4 ints.")

    if any(v < 0 for v in (top, bottom, left, right)):
        raise ValueError("All pad_width values must be non-negative.")

    valid_modes = ("constant", "reflect", "edge")
    if mode not in valid_modes:
        raise ValueError(f"Unknown mode '{mode}'. Choose from {valid_modes}.")

    is_3d = img.ndim == 3

    if mode == "constant":
        pad_spec = ((top, bottom), (left, right))
        if is_3d:
            pad_spec += ((0, 0),)
        return np.pad(img, pad_spec, mode="constant", constant_values=constant_value)

    elif mode == "reflect":
        pad_spec = ((top, bottom), (left, right))
        if is_3d:
            pad_spec += ((0, 0),)
        return np.pad(img, pad_spec, mode="reflect")

    elif mode == "edge":
        pad_spec = ((top, bottom), (left, right))
        if is_3d:
            pad_spec += ((0, 0),)
        return np.pad(img, pad_spec, mode="edge")


def to_float(img):
    """
    Convert image to float64 in [0, 1].

    Parameters
    ----------
    img : np.ndarray
        Image with dtype uint8 (values 0-255) or float (assumed [0,1]).

    Returns
    -------
    np.ndarray
        float64 array in [0, 1].

    Raises
    ------
    TypeError
        If `img` is not a numpy ndarray.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(img).__name__}.")
    if img.dtype == np.uint8:
        return img.astype(np.float64) / 255.0
    return img.astype(np.float64)


def to_uint8(img):
    """
    Convert a float image in [0, 1] to uint8 [0, 255].

    Parameters
    ----------
    img : np.ndarray
        Float image array. Values outside [0, 1] are clipped before conversion.

    Returns
    -------
    np.ndarray
        uint8 array with pixel values in [0, 255].

    Raises
    ------
    TypeError
        If `img` is not a numpy ndarray.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(img).__name__}.")
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)