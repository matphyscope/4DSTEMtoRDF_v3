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

    Memory note: the mean is accumulated directly in float64 from the native
    dtype (``mean(..., dtype=np.float64)``) and only the small 2D result is
    materialized as float. A large 4D cube (e.g. 150x150x256x256) is therefore
    NOT upcast to a full float64 copy — critical to avoid tens of GB of RAM.
    """
    if isinstance(data, DataCube):
        if roi is not None and data.ndim == 4:
            data = data.roi(*roi).data
        else:
            data = data.data
    data = np.asarray(data)                       # keep native dtype (no upcast)
    if data.ndim == 2:
        return np.asarray(data, np.float64)
    if data.ndim == 3:
        return data.mean(0, dtype=np.float64)
    if data.ndim == 4:
        dy, dx = data.shape[-2:]
        if roi is not None:
            y0, y1, x0, x1 = roi
            data = data[y0:y1, x0:x1]
        return data.reshape(-1, dy, dx).mean(0, dtype=np.float64)
    raise ValueError(f"unexpected data ndim={data.ndim}")


def median_pattern(data, roi=None):
    """Median diffraction pattern over all scan positions.

    Unlike :func:`to_pattern` (mean), the per-pixel median rejects outliers —
    X-ray hits, hot frames, occasional Bragg spots from stray crystallites — so
    it is the robust "typical" NBED pattern for an amorphous scan. Computed
    detector-row by detector-row to keep peak memory to one ``(n_pos, Qx)``
    slice instead of sorting the whole cube at once.

    Accepts a DataCube or a raw 3D/4D ndarray. ``roi=(y0,y1,x0,x1)`` restricts a
    4D scan first.
    """
    if isinstance(data, DataCube):
        flat = data.roi(*roi)._flat_patterns() if (roi is not None and data.ndim == 4) \
            else data._flat_patterns()
    else:
        arr = np.asarray(data)
        if arr.ndim == 4 and roi is not None:
            y0, y1, x0, x1 = roi
            arr = arr[y0:y1, x0:x1]
        flat = arr.reshape(-1, *arr.shape[-2:])
    n, Qy, Qx = flat.shape
    out = np.empty((Qy, Qx), np.float64)
    for y in range(Qy):
        out[y] = np.median(np.asarray(flat[:, y, :], np.float64), axis=0)
    return out


def subtract_reference(data, ref, clip_negative=False):
    """Subtract a common 2D reference pattern from every diffraction pattern.

    The classic use is **vacuum / empty-region subtraction**: average the patterns
    over a part of the scan with no sample (e.g. the bottom rows) and subtract that
    from the whole cube. The empty region then goes to ~0, so an unsupervised
    decomposition (PCA/NMF) no longer spends its leading component on the trivial
    sample-vs-vacuum contrast and instead separates *structural* differences within
    the material. Also removes detector background and the common direct-beam tail.

    Returns a new DataCube (float32) — the reference broadcasts over the scan axes,
    so peak memory is one float32 copy. ``clip_negative`` floors the result at 0
    (useful before an NMF that needs non-negative patterns).
    """
    ref = np.asarray(ref, np.float32)
    is_cube = isinstance(data, DataCube)
    arr = data.data if is_cube else np.asarray(data)
    if arr.shape[-2:] != ref.shape:
        raise ValueError(f"reference {ref.shape} != detector {arr.shape[-2:]}")
    out = arr.astype(np.float32) - ref                 # broadcasts (...,Qy,Qx)-(Qy,Qx)
    if clip_negative:
        np.clip(out, 0, None, out=out)
    if is_cube:
        from ..io.datacube import Calibration
        return DataCube(out, calibration=data.calibration,
                        metadata={**data.metadata, "reference_subtracted": True},
                        name=data.name + "-refsub")
    return out


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


def bin_cube_detector(data, factor):
    """Bin the DETECTOR axes of a 3D/4D cube by an integer ``factor``.

    Averages ``factor×factor`` blocks of the last two (detector) axes, leaving
    the scan axes untouched. Cuts the per-pattern pixel count by ``factor²`` —
    essential to fit a spatially-resolved NMF (one row per scan position) of a
    multi-GB 4D cube in memory. Accepts a DataCube or an ndarray; returns the
    same type.
    """
    from ..io.datacube import DataCube

    is_cube = isinstance(data, DataCube)
    arr = data.data if is_cube else np.asarray(data)
    if arr.ndim not in (3, 4):
        raise ValueError(f"bin_cube_detector needs 3D/4D, got {arr.ndim}D")
    *scan, dy, dx = arr.shape
    dy2, dx2 = (dy // factor) * factor, (dx // factor) * factor
    arr = arr[..., :dy2, :dx2]
    new_shape = (*scan, dy2 // factor, factor, dx2 // factor, factor)
    binned = arr.reshape(new_shape).mean(axis=(-3, -1), dtype=np.float64)
    if not is_cube:
        return binned
    from ..io.datacube import Calibration
    cal = data.calibration
    new_cal = Calibration(
        q_per_px=None if cal.q_per_px is None else cal.q_per_px * factor,
        r_per_px=cal.r_per_px, q_unit=cal.q_unit,
        center=None if cal.center is None else (cal.center[0] / factor,
                                                cal.center[1] / factor),
        extra=dict(cal.extra),
    )
    return DataCube(binned, calibration=new_cal, metadata=dict(data.metadata),
                    name=(data.name or "cube") + f"_bin{factor}")


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
