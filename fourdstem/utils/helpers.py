"""
fourdstem.utils.helpers
=======================
Small cross-cutting utilities shared by the rest of the package: numpy
compatibility shims, safe array reductions, and simple geometry helpers.
"""
from __future__ import annotations
import numpy as np

# numpy>=2.0 renamed trapz -> trapezoid; keep both working.
trapezoid = getattr(np, "trapezoid", None) or np.trapz


def as_float_array(a):
    """Return ``a`` as a contiguous float64 ndarray without copying when possible."""
    return np.ascontiguousarray(a, dtype=np.float64)


def nan_safe_normalize(img, lo=1.0, hi=99.0):
    """Scale ``img`` to [0, 1] using robust percentile clipping (ignores NaNs)."""
    img = np.asarray(img, float)
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return np.zeros_like(img)
    vlo, vhi = np.percentile(finite, [lo, hi])
    if vhi <= vlo:
        vhi = vlo + 1e-12
    out = (img - vlo) / (vhi - vlo)
    return np.clip(out, 0.0, 1.0)


def radial_coordinate(shape, center):
    """Radius (in pixels) of every pixel from ``center=(cx, cy)`` for a 2D grid."""
    H, W = shape
    cx, cy = center
    yy, xx = np.mgrid[0:H, 0:W]
    return np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)


def angular_coordinate(shape, center):
    """Azimuthal angle (radians, [-pi, pi]) of every pixel from ``center``."""
    H, W = shape
    cx, cy = center
    yy, xx = np.mgrid[0:H, 0:W]
    return np.arctan2(yy - cy, xx - cx)


def moving_average(x, n):
    """Simple centered moving average with edge padding; ``n`` must be odd-ish."""
    x = np.asarray(x, float)
    if n <= 1:
        return x
    k = np.ones(n) / n
    return np.convolve(x, k, mode="same")
