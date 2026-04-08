"""
minicv.drawing
==============
Drawing primitives on NumPy image arrays.
All functions operate in-place AND return the modified image.
Both grayscale (H, W) and RGB (H, W, 3) images are supported.
Coordinates that fall outside the image boundary are silently clipped.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_image(img):
    if not isinstance(img, np.ndarray):
        raise TypeError(f"img must be np.ndarray, got {type(img).__name__}.")
    if img.ndim not in (2, 3):
        raise ValueError(f"img must be 2D or 3D, got shape {img.shape}.")


def _parse_color(color, img):
    """
    Normalise `color` to match the image channels and dtype.

    For grayscale images a scalar (int/float) is expected.
    For RGB images a 3-element sequence is expected.
    """
    is_rgb = img.ndim == 3
    if is_rgb:
        if not hasattr(color, "__len__") or len(color) != 3:
            raise ValueError(
                "For RGB images, color must be a 3-element sequence (R, G, B)."
            )
        color = tuple(int(c) if img.dtype == np.uint8 else float(c) for c in color)
    else:
        if not isinstance(color, (int, float, np.integer, np.floating)):
            raise ValueError("For grayscale images, color must be a scalar.")
        color = int(color) if img.dtype == np.uint8 else float(color)
    return color


def _set_pixel(img, r, c, color):
    """Set pixel (r, c) if within bounds; silently skip if outside."""
    H, W = img.shape[:2]
    if 0 <= r < H and 0 <= c < W:
        img[r, c] = color


# ---------------------------------------------------------------------------
# Public drawing functions
# ---------------------------------------------------------------------------

def draw_point(img, x, y, color, radius=1):
    """
    Draw a filled circular point (disc) at position (x, y).

    Parameters
    ----------
    img : np.ndarray
        Target image, modified in-place. Shape (H, W) or (H, W, 3).
    x : int
        Column coordinate (horizontal axis).
    y : int
        Row coordinate (vertical axis, 0 = top).
    color : scalar or 3-tuple
        Pixel value. Scalar for grayscale; (R, G, B) for RGB.
        For uint8 images use [0, 255]; for float use [0, 1].
    radius : int, optional
        Disc radius in pixels. Default is 1 (single pixel when radius=0).

    Returns
    -------
    np.ndarray
        The modified image (same object as `img`).

    Raises
    ------
    TypeError
        If `img` is not np.ndarray or `radius` is not int.
    ValueError
        If color format is incompatible with image channels.
    """
    _validate_image(img)
    color = _parse_color(color, img)
    if not isinstance(radius, int) or radius < 0:
        raise TypeError("radius must be a non-negative int.")

    H, W = img.shape[:2]
    r, c = int(y), int(x)

    if radius == 0:
        _set_pixel(img, r, c, color)
    else:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr * dr + dc * dc <= radius * radius:
                    _set_pixel(img, r + dr, c + dc, color)
    return img


def draw_line(img, x1, y1, x2, y2, color, thickness=1):
    """
    Draw an anti-aliased-free line from (x1, y1) to (x2, y2) using
    Bresenham's line algorithm.

    Parameters
    ----------
    img : np.ndarray
        Target image, modified in-place. Shape (H, W) or (H, W, 3).
    x1, y1 : int
        Start column and row.
    x2, y2 : int
        End column and row.
    color : scalar or 3-tuple
        Pixel value compatible with the image dtype and channels.
    thickness : int, optional
        Line thickness in pixels. Default is 1.

    Returns
    -------
    np.ndarray
        The modified image.

    Raises
    ------
    TypeError
        If `img` is not np.ndarray.
    ValueError
        If color is incompatible with image channels.
    """
    _validate_image(img)
    color = _parse_color(color, img)
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    t = max(0, int(thickness))

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    cx, cy = x1, y1
    while True:
        # Draw a small disc for thickness
        for dr in range(-(t // 2), t // 2 + 1):
            for dc in range(-(t // 2), t // 2 + 1):
                _set_pixel(img, cy + dr, cx + dc, color)

        if cx == x2 and cy == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            cx += sx
        if e2 < dx:
            err += dx
            cy += sy

    return img


def draw_rectangle(img, x, y, w, h, color, thickness=1, filled=False):
    """
    Draw a rectangle with top-left corner at (x, y), width w, height h.

    Parameters
    ----------
    img : np.ndarray
        Target image, modified in-place. Shape (H, W) or (H, W, 3).
    x : int
        Column of the top-left corner.
    y : int
        Row of the top-left corner.
    w : int
        Width of the rectangle in pixels (must be ≥ 1).
    h : int
        Height of the rectangle in pixels (must be ≥ 1).
    color : scalar or 3-tuple
        Pixel value compatible with the image dtype and channels.
    thickness : int, optional
        Border thickness in pixels. Ignored when `filled=True`. Default is 1.
    filled : bool, optional
        If True, fill the rectangle interior. Default is False.

    Returns
    -------
    np.ndarray
        The modified image.

    Raises
    ------
    TypeError
        If `img` is not np.ndarray.
    ValueError
        If `w` or `h` < 1, or color is incompatible.
    """
    _validate_image(img)
    color = _parse_color(color, img)
    x, y, w, h = int(x), int(y), int(w), int(h)
    if w < 1 or h < 1:
        raise ValueError(f"Rectangle w and h must be ≥ 1, got w={w}, h={h}.")

    H, W = img.shape[:2]

    if filled:
        r_start = max(0, y)
        r_end   = min(H, y + h)
        c_start = max(0, x)
        c_end   = min(W, x + w)
        if r_start < r_end and c_start < c_end:
            img[r_start:r_end, c_start:c_end] = color
    else:
        t = max(1, int(thickness))
        # Top and bottom edges
        draw_line(img, x,       y,       x + w - 1, y,           color, t)
        draw_line(img, x,       y + h - 1, x + w - 1, y + h - 1, color, t)
        # Left and right edges
        draw_line(img, x,       y,       x,           y + h - 1, color, t)
        draw_line(img, x + w - 1, y,     x + w - 1,   y + h - 1, color, t)

    return img


def draw_polygon(img, points, color, thickness=1, filled=False):
    """
    Draw a polygon defined by a sequence of (x, y) vertices.

    Parameters
    ----------
    img : np.ndarray
        Target image, modified in-place. Shape (H, W) or (H, W, 3).
    points : list of (int, int)
        Ordered list of (x, y) vertex coordinates. At least 2 required.
    color : scalar or 3-tuple
        Pixel value compatible with image dtype and channels.
    thickness : int, optional
        Edge thickness in pixels. Default is 1.
    filled : bool, optional
        If True, fill the polygon interior using scanline fill.
        Default is False.

    Returns
    -------
    np.ndarray
        The modified image.

    Raises
    ------
    TypeError
        If `img` is not np.ndarray or `points` is not a list/tuple.
    ValueError
        If fewer than 2 points are provided, or color is incompatible.
    """
    _validate_image(img)
    color = _parse_color(color, img)
    if not isinstance(points, (list, tuple)):
        raise TypeError("points must be a list or tuple of (x, y) pairs.")
    if len(points) < 2:
        raise ValueError("At least 2 points are required to draw a polygon.")

    pts = [(int(x), int(y)) for x, y in points]

    if filled:
        _fill_polygon(img, pts, color)
    else:
        n = len(pts)
        for i in range(n):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % n]
            draw_line(img, x1, y1, x2, y2, color, thickness)

    return img


def _fill_polygon(img, pts, color):
    """
    Fill a polygon using a scanline algorithm.

    Parameters
    ----------
    img : np.ndarray
        Target image array (modified in-place).
    pts : list of (int, int)
        Vertex coordinates (x, y).
    color : scalar or tuple
        Fill color matching the image channels/dtype.
    """
    H, W = img.shape[:2]
    ys = [p[1] for p in pts]
    y_min = max(0, min(ys))
    y_max = min(H - 1, max(ys))
    n = len(pts)

    for y in range(y_min, y_max + 1):
        intersections = []
        for i in range(n):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % n]
            if y1 == y2:
                continue
            if min(y1, y2) <= y < max(y1, y2):
                # x coordinate of the intersection
                xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                intersections.append(xi)
        intersections.sort()
        for k in range(0, len(intersections) - 1, 2):
            c_start = int(np.ceil(intersections[k]))
            c_end   = int(np.floor(intersections[k + 1]))
            c_start = max(0, c_start)
            c_end   = min(W - 1, c_end)
            if c_start <= c_end:
                img[y, c_start: c_end + 1] = color