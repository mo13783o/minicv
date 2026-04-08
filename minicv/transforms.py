"""
minicv.transforms
=================
Geometric transformations: resize, rotate, translate.
All operations are implemented from scratch using NumPy.
"""

import numpy as np
from .utils import to_float, to_uint8, clip_image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bilinear_interp(img, y_coords, x_coords):
    """
    Vectorised bilinear interpolation on a single-channel float image.

    Parameters
    ----------
    img : np.ndarray
        2-D float array (H, W).
    y_coords : np.ndarray
        Float row coordinates, any shape.
    x_coords : np.ndarray
        Float column coordinates, same shape as y_coords.

    Returns
    -------
    np.ndarray
        Interpolated values, same shape as y_coords / x_coords.
    """
    H, W = img.shape
    y0 = np.floor(y_coords).astype(np.int64)
    x0 = np.floor(x_coords).astype(np.int64)
    y1 = y0 + 1
    x1 = x0 + 1

    # Clip to valid range
    y0c = np.clip(y0, 0, H - 1)
    y1c = np.clip(y1, 0, H - 1)
    x0c = np.clip(x0, 0, W - 1)
    x1c = np.clip(x1, 0, W - 1)

    dy = y_coords - y0
    dx = x_coords - x0

    top_left     = img[y0c, x0c]
    top_right    = img[y0c, x1c]
    bottom_left  = img[y1c, x0c]
    bottom_right = img[y1c, x1c]

    return (
        top_left     * (1 - dy) * (1 - dx)
        + top_right  * (1 - dy) * dx
        + bottom_left  * dy     * (1 - dx)
        + bottom_right * dy     * dx
    )


def _build_output_coords(out_H, out_W):
    """Return meshgrid of (row, col) indices for an output image."""
    rows = np.arange(out_H)
    cols = np.arange(out_W)
    col_grid, row_grid = np.meshgrid(cols, rows)  # both shape (out_H, out_W)
    return row_grid.astype(np.float64), col_grid.astype(np.float64)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resize(image, new_size, method="bilinear"):
    """
    Resize an image to `new_size` using nearest-neighbour or bilinear interpolation.

    Parameters
    ----------
    image : np.ndarray
        Input image of shape (H, W) or (H, W, 3). uint8 or float.
    new_size : tuple of (int, int)
        Target spatial dimensions as (new_height, new_width). Both must be ≥ 1.
    method : str, optional
        Interpolation method. "nearest" or "bilinear". Default is "bilinear".

    Returns
    -------
    np.ndarray
        Resized image of shape (new_H, new_W) or (new_H, new_W, 3),
        same dtype as `image`.

    Raises
    ------
    TypeError
        If `image` is not np.ndarray, or `new_size` is not a tuple/list.
    ValueError
        If `new_size` elements are not positive integers, image has wrong dims,
        or `method` is unknown.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be np.ndarray, got {type(image).__name__}.")
    if not isinstance(new_size, (tuple, list)) or len(new_size) != 2:
        raise TypeError("new_size must be a (height, width) tuple.")
    new_H, new_W = int(new_size[0]), int(new_size[1])
    if new_H < 1 or new_W < 1:
        raise ValueError(f"new_size dimensions must be ≥ 1, got ({new_H}, {new_W}).")
    if image.ndim not in (2, 3):
        raise ValueError(f"image must be 2D or 3D, got shape {image.shape}.")
    if method not in ("nearest", "bilinear"):
        raise ValueError(f"method must be 'nearest' or 'bilinear', got '{method}'.")

    original_dtype = image.dtype
    img_f = to_float(image)
    src_H, src_W = img_f.shape[:2]

    # Map output pixel centres → input pixel coordinates
    row_out, col_out = _build_output_coords(new_H, new_W)
    # +0.5 centre-alignment, then scale
    src_row = (row_out + 0.5) * (src_H / new_H) - 0.5
    src_col = (col_out + 0.5) * (src_W / new_W) - 0.5

    def _resize_channel(ch):
        if method == "nearest":
            r = np.clip(np.round(src_row).astype(np.int64), 0, src_H - 1)
            c = np.clip(np.round(src_col).astype(np.int64), 0, src_W - 1)
            return ch[r, c]
        else:  # bilinear
            return _bilinear_interp(ch, src_row, src_col)

    if img_f.ndim == 2:
        result = _resize_channel(img_f)
    else:
        result = np.stack(
            [_resize_channel(img_f[:, :, c]) for c in range(img_f.shape[2])],
            axis=-1,
        )

    result = np.clip(result, 0.0, 1.0)
    if original_dtype == np.uint8:
        return to_uint8(result)
    return result.astype(original_dtype)


def rotate(image, angle, expand=False):
    """
    Rotate an image counter-clockwise by `angle` degrees about its centre.

    Uses an inverse-mapping strategy with bilinear interpolation so every
    output pixel is sampled exactly once. Pixels outside the original canvas
    are filled with 0 (black).

    Parameters
    ----------
    image : np.ndarray
        Input image of shape (H, W) or (H, W, 3). uint8 or float.
    angle : float
        Rotation angle in degrees. Positive = counter-clockwise.
    expand : bool, optional
        If True, the output canvas is enlarged to contain the full rotated image
        without cropping. Default is False (same canvas size as input).

    Returns
    -------
    np.ndarray
        Rotated image, same dtype as `image`.
        Shape is (H, W[, 3]) when expand=False, or enlarged when expand=True.

    Raises
    ------
    TypeError
        If `image` is not np.ndarray or `angle` is not numeric.
    ValueError
        If `image` is not 2D or 3D.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be np.ndarray, got {type(image).__name__}.")
    if not isinstance(angle, (int, float)):
        raise TypeError(f"angle must be numeric, got {type(angle).__name__}.")
    if image.ndim not in (2, 3):
        raise ValueError(f"image must be 2D or 3D, got shape {image.shape}.")

    original_dtype = image.dtype
    img_f = to_float(image)
    src_H, src_W = img_f.shape[:2]

    theta = np.deg2rad(angle)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    if expand:
        # Compute bounding box of rotated image
        corners = np.array([
            [0,       0],
            [src_W,   0],
            [0,       src_H],
            [src_W,   src_H],
        ], dtype=np.float64)
        cx, cy = src_W / 2.0, src_H / 2.0
        corners -= np.array([cx, cy])
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        rotated_corners = corners @ R.T
        min_x = rotated_corners[:, 0].min()
        max_x = rotated_corners[:, 0].max()
        min_y = rotated_corners[:, 1].min()
        max_y = rotated_corners[:, 1].max()
        out_W = int(np.ceil(max_x - min_x))
        out_H = int(np.ceil(max_y - min_y))
        out_cx, out_cy = out_W / 2.0, out_H / 2.0
    else:
        out_H, out_W = src_H, src_W
        out_cx, out_cy = src_W / 2.0, src_H / 2.0

    src_cx, src_cy = src_W / 2.0, src_H / 2.0

    # Inverse rotation: output → source
    row_out, col_out = _build_output_coords(out_H, out_W)
    # Centre output coords
    dx = col_out - out_cx
    dy = row_out - out_cy
    # Apply inverse rotation (transpose of R)
    src_col = cos_t * dx + sin_t * dy + src_cx
    src_row = -sin_t * dx + cos_t * dy + src_cy

    # Mask for out-of-bounds pixels
    valid = (
        (src_row >= 0) & (src_row <= src_H - 1) &
        (src_col >= 0) & (src_col <= src_W - 1)
    )

    def _rotate_channel(ch):
        out = np.zeros((out_H, out_W), dtype=np.float64)
        out[valid] = _bilinear_interp(ch, src_row, src_col)[valid]
        return out

    if img_f.ndim == 2:
        result = _rotate_channel(img_f)
    else:
        result = np.stack(
            [_rotate_channel(img_f[:, :, c]) for c in range(img_f.shape[2])],
            axis=-1,
        )

    result = np.clip(result, 0.0, 1.0)
    if original_dtype == np.uint8:
        return to_uint8(result)
    return result.astype(original_dtype)


def translate(image, tx, ty):
    """
    Translate (shift) an image by (tx, ty) pixels.

    Pixels that shift out of the canvas are discarded; vacated regions are
    filled with 0 (black). The output has the same shape as the input.

    Parameters
    ----------
    image : np.ndarray
        Input image of shape (H, W) or (H, W, 3). uint8 or float.
    tx : int or float
        Horizontal shift in pixels. Positive → shift right.
    ty : int or float
        Vertical shift in pixels. Positive → shift down.

    Returns
    -------
    np.ndarray
        Translated image, same shape and dtype as `image`.

    Raises
    ------
    TypeError
        If `image` is not np.ndarray, or `tx`/`ty` are not numeric.
    ValueError
        If `image` is not 2D or 3D.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be np.ndarray, got {type(image).__name__}.")
    for name, val in (("tx", tx), ("ty", ty)):
        if not isinstance(val, (int, float)):
            raise TypeError(f"{name} must be numeric, got {type(val).__name__}.")
    if image.ndim not in (2, 3):
        raise ValueError(f"image must be 2D or 3D, got shape {image.shape}.")

    original_dtype = image.dtype
    img_f = to_float(image)
    H, W = img_f.shape[:2]

    row_out, col_out = _build_output_coords(H, W)
    src_row = row_out - ty
    src_col = col_out - tx

    valid = (
        (src_row >= 0) & (src_row <= H - 1) &
        (src_col >= 0) & (src_col <= W - 1)
    )

    def _translate_channel(ch):
        out = np.zeros((H, W), dtype=np.float64)
        out[valid] = _bilinear_interp(ch, src_row, src_col)[valid]
        return out

    if img_f.ndim == 2:
        result = _translate_channel(img_f)
    else:
        result = np.stack(
            [_translate_channel(img_f[:, :, c]) for c in range(img_f.shape[2])],
            axis=-1,
        )

    result = np.clip(result, 0.0, 1.0)
    if original_dtype == np.uint8:
        return to_uint8(result)
    return result.astype(original_dtype)