"""
fourdstem.preprocess.clean
==========================
Detector cleanup: hot/dead pixel removal and light denoising. These operate on
a single 2D diffraction pattern and return a cleaned copy, so they slot in
before centering / integration.

    remove_hot_pixels   replace outlier-bright pixels with a local median
    remove_dead_pixels  replace zero/negative (dead) pixels with a local median
    median_denoise      plain median filter (small kernel)
    clean_pattern       convenience: hot + dead in one call
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import median_filter


def remove_hot_pixels(img, threshold=8.0, size=3, min_fraction=0.02,
                      return_mask=False):
    """Replace hot (outlier-bright) pixels with the local median.

    A pixel is "hot" if it exceeds the local median by more than ``threshold``
    times the robust (MAD-based) spread of the local-median residual AND by more
    than ``min_fraction`` of the image's robust dynamic range. The second,
    absolute condition prevents over-flagging smooth gradients (e.g. ring
    flanks) when the MAD collapses toward zero on a clean image.

    Parameters
    ----------
    img : 2D array
    threshold : float
        Sensitivity in robust-sigma units. Lower = more aggressive.
    size : int
        Median-filter kernel size (odd).
    min_fraction : float
        Absolute floor as a fraction of ``percentile(img, 99.5) - median(img)``.
    return_mask : bool
        If True, also return the boolean hot-pixel mask.

    Returns
    -------
    cleaned : 2D array
    mask : 2D bool array  (only if ``return_mask``)
    """
    img = np.asarray(img, float)
    med = median_filter(img, size=size)
    resid = img - med
    mad = np.median(np.abs(resid - np.median(resid))) + 1e-12
    robust_std = 1.4826 * mad
    dyn = np.percentile(img, 99.5) - np.median(img)
    floor = min_fraction * max(dyn, 1e-12)
    hot = (resid > threshold * robust_std) & (resid > floor)
    cleaned = np.where(hot, med, img)
    if return_mask:
        return cleaned, hot
    return cleaned


def remove_dead_pixels(img, size=3, return_mask=False):
    """Replace dead pixels (<= 0) with the local median of finite neighbours."""
    img = np.asarray(img, float)
    dead = ~np.isfinite(img) | (img <= 0)
    if not dead.any():
        return (img.copy(), dead) if return_mask else img.copy()
    filled = np.where(dead, np.nan, img)
    med = median_filter(np.nan_to_num(filled, nan=np.nanmedian(filled)), size=size)
    cleaned = np.where(dead, med, img)
    if return_mask:
        return cleaned, dead
    return cleaned


def median_denoise(img, size=3):
    """Small-kernel median filter (removes salt-and-pepper noise)."""
    return median_filter(np.asarray(img, float), size=size)


def clean_pattern(img, hot_threshold=8.0, size=3, do_dead=True):
    """Convenience: remove dead pixels then hot pixels in one call."""
    out = np.asarray(img, float)
    if do_dead:
        out = remove_dead_pixels(out, size=size)
    out = remove_hot_pixels(out, threshold=hot_threshold, size=size)
    return out
