"""
fourdstem.preprocess.transform
==============================
Shape/geometry transforms on cubes and patterns: collapse a cube to a single
diffraction pattern, crop/bin the detector, and build a polar (r, θ)
representation.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import map_coordinates

from ..io.datacube import DataCube


def to_pattern(data, roi=None):
    """Collapse array data to a single 2D diffraction pattern (mean over scan).

    Accepts a DataCube or a raw ndarray (2D/3D/4D). ``roi=(y0,y1,x0,x1)``
    restricts a 4D scan before averaging.
    """
    if isinstance(data, DataCube):
        if roi is not None and data.ndim == 4:
            data = data.roi(*roi).data
        else:
            data = data.data
    data = np.asarray(data, float)
    if data.ndim == 2:
        return data
    if data.ndim == 3:
        return data.mean(0)
    if data.ndim == 4:
        sy, sx, dy, dx = data.shape
        if roi is not None:
            y0, y1, x0, x1 = roi
            data = data[y0:y1, x0:x1]
        return data.reshape(-1, dy, dx).mean(0)
    raise ValueError(f"unexpected data ndim={data.ndim}")


def crop_detector(img, center, half_size):
    """Crop a square window of half-width ``half_size`` around ``center``.

    Returns ``(cropped, new_center)``. Clamps to the image bounds.
    """
    img = np.asarray(img, float)
    H, W = img.shape
    cx, cy = center
    x0 = max(0, int(round(cx - half_size)))
    x1 = min(W, int(round(cx + half_size)))
    y0 = max(0, int(round(cy - half_size)))
    y1 = min(H, int(round(cy + half_size)))
    return img[y0:y1, x0:x1], (cx - x0, cy - y0)


def bin_detector(img, factor):
    """Bin a 2D pattern by an integer ``factor`` (mean over factor×factor blocks)."""
    img = np.asarray(img, float)
    H, W = img.shape
    Hc, Wc = (H // factor) * factor, (W // factor) * factor
    img = img[:Hc, :Wc]
    return img.reshape(Hc // factor, factor, Wc // factor, factor).mean((1, 3))


def polar_transform(img, center, n_r=None, n_theta=360, r_max=None, mask=None):
    """Resample a Cartesian pattern onto a polar grid ``(theta, r)``.

    Returns ``(polar, r_axis, theta_axis)``. Masked/off-image pixels are NaN.
    Handy for visualizing rings and for azimuthal-variance / texture analysis.
    """
    img = np.asarray(img, float)
    H, W = img.shape
    cx, cy = center
    if r_max is None:
        r_max = min(cx, cy, W - cx, H - cy)
    if n_r is None:
        n_r = int(round(r_max))
    r_axis = np.linspace(0, r_max, n_r)
    theta_axis = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    TH, R = np.meshgrid(theta_axis, r_axis, indexing="ij")
    X = cx + R * np.cos(TH)
    Y = cy + R * np.sin(TH)
    polar = map_coordinates(img, [Y.ravel(), X.ravel()], order=1,
                            mode="constant", cval=np.nan).reshape(TH.shape)
    if mask is not None:
        mm = map_coordinates(mask.astype(float), [Y.ravel(), X.ravel()],
                             order=0, mode="constant", cval=1).reshape(TH.shape)
        polar[mm > 0.5] = np.nan
    return polar, r_axis, theta_axis
