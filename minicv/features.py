"""
minicv.features
===============
Image feature extraction: histograms, histogram equalisation, Sobel features.
"""

import numpy as np
from .utils import to_float, to_uint8
from .filters import sobel_gradients


def compute_histogram(image, bins=256, channel=None):
    """
    Compute the intensity histogram of a grayscale or RGB image.

    Parameters
    ----------
    image : np.ndarray
        Grayscale (H, W) or RGB (H, W, 3). uint8 or float.
    bins : int, optional
        Number of histogram bins. Default is 256.
    channel : int or None, optional
        For RGB images: 0=R, 1=G, 2=B selects a single channel.
        None → for grayscale returns a single histogram; for RGB returns
        a dict with keys "r", "g", "b", each containing (counts, bin_edges).
        Default is None.

    Returns
    -------
    For grayscale or when `channel` is specified:
        tuple (counts, bin_edges) — same semantics as np.histogram.
    For RGB with channel=None:
        dict {"r": (counts, edges), "g": (counts, edges), "b": (counts, edges)}.

    Raises
    ------
    TypeError
        If `image` is not np.ndarray or `bins` is not int.
    ValueError
        If `image` is not 2D/3D, `bins` < 1, or `channel` is out of range.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be np.ndarray, got {type(image).__name__}.")
    if not isinstance(bins, int) or bins < 1:
        raise TypeError(f"bins must be a positive int, got {bins!r}.")
    if image.ndim not in (2, 3):
        raise ValueError(f"image must be 2D or 3D, got shape {image.shape}.")

    img_f = to_float(image)

    def _hist(arr):
       hist, _ = np.histogram(arr.ravel(), bins=bins, range=(0.0, 1.0))
       return hist
    if img_f.ndim == 2:
        return _hist(img_f)

    # RGB path
    if channel is not None:
        if channel not in (0, 1, 2):
            raise ValueError(f"channel must be 0, 1, or 2, got {channel}.")
        return _hist(img_f[:, :, channel])

    return {
        "r": _hist(img_f[:, :, 0]),
        "g": _hist(img_f[:, :, 1]),
        "b": _hist(img_f[:, :, 2]),
    }


def histogram_equalization(image):
    """
    Enhance contrast via histogram equalisation.

    For grayscale images the standard CDF-based equalisation is applied.
    For RGB images, equalisation is applied independently to each channel,
    which may shift hues — for hue-preserving equalisation convert to
    HSV first (outside this library's scope).

    Parameters
    ----------
    image : np.ndarray
        Grayscale (H, W) or RGB (H, W, 3). uint8 or float.

    Returns
    -------
    np.ndarray
        Contrast-enhanced image, same shape and dtype as `image`.

    Raises
    ------
    TypeError
        If `image` is not np.ndarray.
    ValueError
        If `image` is not 2D or 3D.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be np.ndarray, got {type(image).__name__}.")
    if image.ndim not in (2, 3):
        raise ValueError(f"image must be 2D or 3D, got shape {image.shape}.")

    original_dtype = image.dtype

    # Work in uint8 space [0-255] for clean bin alignment
    if image.dtype != np.uint8:
        img_u8 = to_uint8(to_float(image))
    else:
        img_u8 = image.copy()

    def _equalize_channel(ch):
        hist, _ = np.histogram(ch.ravel(), bins=256, range=(0, 256))
        cdf = hist.cumsum()
        # Normalise CDF ignoring zero bins
        cdf_min = cdf[cdf > 0].min()
        total = ch.size
        lut = np.round(
            (cdf - cdf_min) / (total - cdf_min) * 255
        ).clip(0, 255).astype(np.uint8)
        return lut[ch]

    if img_u8.ndim == 2:
        result = _equalize_channel(img_u8)
    else:
        result = np.stack(
            [_equalize_channel(img_u8[:, :, c]) for c in range(3)],
            axis=-1,
        )

    if original_dtype != np.uint8:
        return to_float(result).astype(original_dtype)
    return result


def sobel_features(image):
    """
    Extract Sobel-based edge features from an image.

    Wraps `filters.sobel_gradients` and returns a structured feature map
    dictionary suitable for downstream processing.

    Parameters
    ----------
    image : np.ndarray
        Grayscale (H, W) or RGB (H, W, 3). uint8 or float.
        RGB images are converted to grayscale internally.

    Returns
    -------
    dict with keys:
        "magnitude"      : np.ndarray (H, W), float64 in [0, 1].
        "angle"          : np.ndarray (H, W), float64 in degrees [-180, 180].
        "gx"             : np.ndarray (H, W), float64 horizontal gradient.
        "gy"             : np.ndarray (H, W), float64 vertical gradient.
        "edge_binary"    : np.ndarray (H, W), bool — magnitude > threshold.
        "threshold_used" : float — Otsu-inspired threshold applied.

    Raises
    ------
    TypeError
        If `image` is not np.ndarray.
    ValueError
        If `image` is not 2D or 3D.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be np.ndarray, got {type(image).__name__}.")
    if image.ndim not in (2, 3):
        raise ValueError(f"image must be 2D or 3D, got shape {image.shape}.")

    grads = sobel_gradients(image)
    mag = grads["magnitude"]

    # Simple Otsu-inspired threshold on the magnitude histogram
    threshold = _otsu_threshold(mag)
    edge_binary = mag > threshold

    return {
        "magnitude":      mag,
        "angle":          grads["angle"],
        "gx":             grads["gx"],
        "gy":             grads["gy"],
        "edge_binary":    edge_binary,
        "threshold_used": threshold,
    }


def _otsu_threshold(img_float):
    """
    Compute Otsu's threshold for a normalised [0,1] image.

    Parameters
    ----------
    img_float : np.ndarray
        Float array in [0, 1].

    Returns
    -------
    float
        Threshold value in [0, 1].
    """
    bins = 256
    hist, bin_edges = np.histogram(img_float.ravel(), bins=bins, range=(0.0, 1.0))
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    hist = hist.astype(np.float64)
    total = hist.sum()

    best_thresh = 0.0
    best_var = -1.0
    weight1_cum = 0.0
    mean1_cum = 0.0

    total_mean = (hist * bin_centres).sum() / total

    for i in range(bins):
        weight1_cum += hist[i]
        weight2 = total - weight1_cum
        if weight1_cum == 0 or weight2 == 0:
            continue
        mean1_cum += hist[i] * bin_centres[i]
        mean1 = mean1_cum / weight1_cum
        mean2 = (total_mean * total - mean1_cum) / weight2
        between_var = (weight1_cum * weight2) * (mean1 - mean2) ** 2
        if between_var > best_var:
            best_var = between_var
            best_thresh = bin_centres[i]

    return best_thresh