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


def bad_pixel_map(reference, hot_threshold=8.0, max_frac=0.01):
    """Fixed hot/dead pixel mask from a reference pattern (mean or max over scan).

    Hot = isolated single-pixel spikes far above the 3x3 local median (X-ray-
    damaged / stuck-bright detector pixels), NOT extended structure like the beam
    or rings. Dead = zero. Detector defects sit at the SAME location every frame,
    so a scan-average does not remove them — detect once here, repair everywhere
    with :func:`repair_bad_pixels`. ``max_frac`` caps the flagged fraction (keeps
    only the most extreme) so real structure is never mass-flagged.
    """
    from scipy.ndimage import median_filter
    ref = np.asarray(reference, float)
    med = median_filter(ref, size=3)           # small kernel -> isolates spikes
    resid = ref - med
    s = np.median(np.abs(resid)) * 1.4826 + 1e-9
    score = resid / s
    hot = score > hot_threshold
    if hot.mean() > max_frac:                  # too many -> keep only the extremes
        cut = np.quantile(score, 1.0 - max_frac)
        hot = score > max(cut, hot_threshold)
    dead = (ref <= 0) & (med > hot_threshold * s)   # zero only WHERE neighbours are bright
    return hot | dead


def repair_bad_pixels(cube, bad_mask):
    """Replace fixed bad pixels in every pattern with their good-neighbour mean.

    Fast and memory-light: loops only over the (few) bad-pixel locations and
    averages each one's in-bounds, non-bad 8-neighbours across the whole scan at
    once. Returns a new DataCube (or ndarray) with the defects repaired.
    """
    from ..io.datacube import DataCube
    is_cube = isinstance(cube, DataCube)
    data = np.array(cube.data if is_cube else cube, dtype=np.float32)  # writable copy
    dy, dx = data.shape[-2:]
    flat = data.reshape(-1, dy, dx)
    bad = np.asarray(bad_mask, bool)
    ys, xs = np.where(bad)
    offs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    for y, x in zip(ys, xs):
        acc = None; cnt = 0
        for oy, ox in offs:
            ny, nx = y + oy, x + ox
            if 0 <= ny < dy and 0 <= nx < dx and not bad[ny, nx]:
                v = flat[:, ny, nx]
                acc = v.astype(np.float64) if acc is None else acc + v
                cnt += 1
        if cnt:
            flat[:, y, x] = (acc / cnt).astype(flat.dtype)
    out = flat.reshape(data.shape)
    if is_cube:
        return DataCube(out, calibration=cube.calibration,
                        metadata={**cube.metadata, "bad_pixels_repaired": int(bad.sum())},
                        name=cube.name)
    return out
