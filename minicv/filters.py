"""
minicv.filters
==============
Image filtering and convolution from scratch using NumPy.
No image-processing libraries are used.
"""

import numpy as np
from .utils import pad_image, to_float, to_uint8


# ---------------------------------------------------------------------------
# Core convolution engine
# ---------------------------------------------------------------------------

def convolve2d(image, kernel, padding="same"):
    """
    Perform a 2-D discrete convolution of a single-channel image with a kernel.

    Uses vectorised sliding-window views (stride tricks) — no explicit
    pixel loops. For multi-channel images, call per-channel and re-stack.

    Parameters
    ----------
    image : np.ndarray
        2-D float array of shape (H, W).
    kernel : np.ndarray
        2-D array of odd spatial dimensions (kH, kW).
    padding : str, optional
        "same"  — output has same spatial size as input (reflect padding).
        "valid" — no padding; output shrinks by (kH-1, kW-1).
        Default is "same".

    Returns
    -------
    np.ndarray
        Convolved float64 array. Shape equals input shape when padding="same".

    Raises
    ------
    TypeError
        If `image` or `kernel` are not np.ndarray.
    ValueError
        If `image` is not 2D, `kernel` is not 2D, kernel dimensions are even,
        or `padding` is not "same" / "valid".
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be np.ndarray, got {type(image).__name__}.")
    if not isinstance(kernel, np.ndarray):
        raise TypeError(f"kernel must be np.ndarray, got {type(kernel).__name__}.")
    if image.ndim != 2:
        raise ValueError(f"image must be 2D, got shape {image.shape}.")
    if kernel.ndim != 2:
        raise ValueError(f"kernel must be 2D, got shape {kernel.shape}.")
    if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise ValueError(
            f"Kernel dimensions must be odd, got {kernel.shape}."
        )
    if padding not in ("same", "valid"):
        raise ValueError(f"padding must be 'same' or 'valid', got '{padding}'.")

    image = image.astype(np.float64)
    kernel = kernel.astype(np.float64)
    kH, kW = kernel.shape

    if padding == "same":
        ph, pw = kH // 2, kW // 2
        image = pad_image(image, (ph, ph, pw, pw), mode="reflect")

    H, W = image.shape
    out_H = H - kH + 1
    out_W = W - kW + 1

    # Build a view of shape (out_H, out_W, kH, kW) using stride tricks
    shape = (out_H, out_W, kH, kW)
    strides = (
        image.strides[0],
        image.strides[1],
        image.strides[0],
        image.strides[1],
    )
    windows = np.lib.stride_tricks.as_strided(image, shape=shape, strides=strides)

    # Flip kernel for true convolution (cross-correlation if not flipped)
    kernel_flipped = kernel[::-1, ::-1]

    # Vectorised dot product over kernel dims
    return np.einsum("ijkl,kl->ij", windows, kernel_flipped)


def apply_filter(image, kernel):
    """
    Apply a 2-D convolution kernel to a grayscale or RGB image.

    Internally calls `convolve2d` per channel and clips the result to [0, 1].

    Parameters
    ----------
    image : np.ndarray
        Grayscale (H, W) or RGB (H, W, 3). uint8 or float.
    kernel : np.ndarray
        2-D kernel array with odd dimensions.

    Returns
    -------
    np.ndarray
        Filtered image, same shape and dtype as `image`, clipped to valid range.

    Raises
    ------
    TypeError
        If `image` or `kernel` are not np.ndarray.
    ValueError
        If image is not 2D/3D, or kernel has even dimensions.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be np.ndarray, got {type(image).__name__}.")
    if not isinstance(kernel, np.ndarray):
        raise TypeError(f"kernel must be np.ndarray, got {type(kernel).__name__}.")
    if image.ndim not in (2, 3):
        raise ValueError(f"image must be 2D or 3D, got shape {image.shape}.")

    original_dtype = image.dtype
    img_f = to_float(image)

    if img_f.ndim == 2:
        result = convolve2d(img_f, kernel, padding="same")
    else:
        channels = [
            convolve2d(img_f[:, :, c], kernel, padding="same")
            for c in range(img_f.shape[2])
        ]
        result = np.stack(channels, axis=-1)

    result = np.clip(result, 0.0, 1.0)

    if original_dtype == np.uint8:
        return to_uint8(result)
    return result.astype(original_dtype)


# ---------------------------------------------------------------------------
# Named filters
# ---------------------------------------------------------------------------

def mean_filter(image, kernel_size=3):
    """
    Smooth an image using a box (mean/averaging) filter.

    Parameters
    ----------
    image : np.ndarray
        Grayscale (H, W) or RGB (H, W, 3). uint8 or float.
    kernel_size : int, optional
        Side length of the square kernel. Must be a positive odd integer.
        Default is 3.

    Returns
    -------
    np.ndarray
        Smoothed image, same shape and dtype as `image`.

    Raises
    ------
    TypeError
        If `image` is not np.ndarray or `kernel_size` is not int.
    ValueError
        If `kernel_size` is even or ≤ 0.
    """
    if not isinstance(kernel_size, int):
        raise TypeError(f"kernel_size must be int, got {type(kernel_size).__name__}.")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}.")

    kernel = np.ones((kernel_size, kernel_size), dtype=np.float64) / (kernel_size ** 2)
    return apply_filter(image, kernel)


def _gaussian_kernel(kernel_size, sigma):
    """Generate a normalised 2-D Gaussian kernel."""
    k = kernel_size // 2
    xs = np.arange(-k, k + 1, dtype=np.float64)
    gauss_1d = np.exp(-(xs ** 2) / (2 * sigma ** 2))
    gauss_2d = np.outer(gauss_1d, gauss_1d)
    return gauss_2d / gauss_2d.sum()


def gaussian_filter(image, kernel_size=5, sigma=1.0):
    """
    Smooth an image using a Gaussian filter.

    The Gaussian kernel is generated analytically (no external libraries).

    Parameters
    ----------
    image : np.ndarray
        Grayscale (H, W) or RGB (H, W, 3). uint8 or float.
    kernel_size : int, optional
        Side length of the square kernel. Must be a positive odd integer.
        Default is 5.
    sigma : float, optional
        Standard deviation of the Gaussian distribution. Must be > 0.
        Default is 1.0.

    Returns
    -------
    np.ndarray
        Gaussian-smoothed image, same shape and dtype as `image`.

    Raises
    ------
    TypeError
        If `image` is not np.ndarray, or `kernel_size` not int, or `sigma` not numeric.
    ValueError
        If `kernel_size` is even or ≤ 0, or `sigma` ≤ 0.
    """
    if not isinstance(kernel_size, int):
        raise TypeError(f"kernel_size must be int, got {type(kernel_size).__name__}.")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}.")
    if not isinstance(sigma, (int, float)):
        raise TypeError(f"sigma must be numeric, got {type(sigma).__name__}.")
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}.")

    kernel = _gaussian_kernel(kernel_size, sigma)
    return apply_filter(image, kernel)


def median_filter(image, kernel_size=3):
    """
    Reduce impulse (salt-and-pepper) noise using a median filter.

    Because the median is a non-linear operation it cannot be expressed as a
    convolution; a controlled sliding-window loop is used instead.
    NumPy advanced indexing keeps the inner body vectorised per window row.

    Parameters
    ----------
    image : np.ndarray
        Grayscale (H, W) or RGB (H, W, 3). uint8 or float.
    kernel_size : int, optional
        Side length of the square neighbourhood. Must be a positive odd integer.
        Default is 3.

    Returns
    -------
    np.ndarray
        Median-filtered image, same shape and dtype as `image`.

    Raises
    ------
    TypeError
        If `image` is not np.ndarray or `kernel_size` is not int.
    ValueError
        If `kernel_size` is even or ≤ 0, or image is not 2D/3D.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be np.ndarray, got {type(image).__name__}.")
    if not isinstance(kernel_size, int):
        raise TypeError(f"kernel_size must be int, got {type(kernel_size).__name__}.")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}.")
    if image.ndim not in (2, 3):
        raise ValueError(f"image must be 2D or 3D, got shape {image.shape}.")

    original_dtype = image.dtype
    img_f = to_float(image)
    k = kernel_size // 2

    def _median_channel(channel):
        padded = pad_image(channel, k, mode="edge")
        H, W = channel.shape
        out = np.empty((H, W), dtype=np.float64)
        for i in range(H):
            for j in range(W):
                patch = padded[i: i + kernel_size, j: j + kernel_size]
                out[i, j] = np.median(patch)
        return out

    if img_f.ndim == 2:
        result = _median_channel(img_f)
    else:
        result = np.stack(
            [_median_channel(img_f[:, :, c]) for c in range(img_f.shape[2])],
            axis=-1,
        )

    result = np.clip(result, 0.0, 1.0)
    if original_dtype == np.uint8:
        return to_uint8(result)
    return result.astype(original_dtype)


def sobel_gradients(image):
    """
    Compute Sobel edge gradients (Gx, Gy, magnitude, angle) for a grayscale image.

    Uses the standard 3×3 Sobel operators:
        Kx = [[-1,0,1],[-2,0,2],[-1,0,1]]
        Ky = [[-1,-2,-1],[0,0,0],[1,2,1]]

    Parameters
    ----------
    image : np.ndarray
        Grayscale image of shape (H, W). uint8 or float.
        If an RGB image is passed it is averaged to grayscale automatically.

    Returns
    -------
    dict with keys:
        "gx"        : np.ndarray (H, W) — horizontal gradient (float64).
        "gy"        : np.ndarray (H, W) — vertical gradient (float64).
        "magnitude" : np.ndarray (H, W) — gradient magnitude, normalised [0,1].
        "angle"     : np.ndarray (H, W) — gradient angle in degrees [-180, 180].

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

    if image.ndim == 3:
        weights = np.array([0.299, 0.587, 0.114])
        image = (image.astype(np.float64) @ weights)
    else:
        image = image.astype(np.float64)

    # Normalise to [0,1] if uint8
    if image.max() > 1.0:
        image = image / 255.0

    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float64)
    Ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float64)

    gx = convolve2d(image, Kx, padding="same")
    gy = convolve2d(image, Ky, padding="same")
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    mag_max = magnitude.max()
    if mag_max > 0:
        magnitude = magnitude / mag_max
    angle = np.degrees(np.arctan2(gy, gx))

    return {"gx": gx, "gy": gy, "magnitude": magnitude, "angle": angle}