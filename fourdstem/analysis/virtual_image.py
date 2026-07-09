"""
fourdstem.analysis.virtual_image
================================
Form real-space images from a 4D-STEM cube by integrating each diffraction
pattern over a virtual detector, plus center-of-mass / DPC-style maps.

All functions take a :class:`~fourdstem.io.datacube.DataCube` and return a
2D image over the scan grid (or 1D for a 3D stack).
"""
from __future__ import annotations
import numpy as np

from ..preprocess.masks import disk_mask, annular_mask
from ..utils.helpers import radial_coordinate


def virtual_image(cube, mask):
    """Integrate each pattern over a boolean detector ``mask`` (True = keep)."""
    return cube.get_virtual_image(mask)


def bright_field(cube, center=None, radius=None):
    """Bright-field image: integrate a central disk detector.

    ``center`` defaults to the cube's calibrated center or the detector middle;
    ``radius`` defaults to 1/8 of the detector.
    """
    dp = cube.dp_shape
    center = _resolve_center(cube, center)
    if radius is None:
        radius = min(dp) / 8.0
    return cube.get_virtual_image(disk_mask(dp, center, radius))


def annular_dark_field(cube, center=None, r_inner=None, r_outer=None):
    """ADF image: integrate an annular detector."""
    dp = cube.dp_shape
    center = _resolve_center(cube, center)
    if r_inner is None:
        r_inner = min(dp) / 6.0
    if r_outer is None:
        r_outer = min(dp) / 2.0
    return cube.get_virtual_image(annular_mask(dp, center, r_inner, r_outer))


def structural_map(cube, center=None, r_inner=None, r_outer=None,
                   norm_inner=None, norm_outer=None):
    """Brightness-normalized structural virtual image: ring-DF / total.

    Integrates a structural annulus (``r_inner..r_outer``, e.g. the amorphous
    ring / FSDP) and divides by the total structural intensity at each scan
    position. This CANCELS per-position brightness variation (dose, thickness,
    scan-line drift — which otherwise show as horizontal stripes) and leaves the
    *structural* contrast, so a phase/interface with different scattering shows
    up as a real-space feature (e.g. a band). Returns a map over the scan grid.
    """
    center = _resolve_center(cube, center)
    dp = cube.dp_shape
    if r_inner is None:
        r_inner = min(dp) / 5.0
    if r_outer is None:
        r_outer = min(dp) / 3.0
    if norm_inner is None:
        norm_inner = min(dp) / 12.0
    if norm_outer is None:
        norm_outer = min(dp) / 2.0
    ring = cube.get_virtual_image(annular_mask(dp, center, r_inner, r_outer))
    total = cube.get_virtual_image(annular_mask(dp, center, norm_inner, norm_outer))
    total = np.where(np.asarray(total) > 0, total, 1.0)
    return np.asarray(ring) / total


def center_of_mass_map(cube, center=None, mask=None, normalize=True):
    """Per-scan-position center-of-mass of the diffraction pattern (DPC).

    Returns ``(com_x, com_y)`` maps, each shaped like the scan grid, giving the
    beam-shift components. Subtracting a reference (mean) yields a DPC signal.
    """
    center = _resolve_center(cube, center)
    dp = cube.dp_shape
    yy, xx = np.mgrid[0:dp[0], 0:dp[1]].astype(float)
    if mask is not None:
        keep = np.asarray(mask, bool)
    else:
        keep = np.ones(dp, bool)

    flat = cube._flat_patterns().astype(float)
    w = flat * keep
    tot = w.sum(axis=(1, 2))
    tot_safe = np.where(tot > 0, tot, 1.0)
    comx = (w * xx).sum(axis=(1, 2)) / tot_safe
    comy = (w * yy).sum(axis=(1, 2)) / tot_safe
    if normalize:
        comx = comx - center[0]
        comy = comy - center[1]
    scan = cube.scan_shape
    if scan:
        comx = comx.reshape(scan)
        comy = comy.reshape(scan)
    return comx, comy


def _resolve_center(cube, center):
    if center is not None:
        return center
    if cube.calibration.center is not None:
        return cube.calibration.center
    qy, qx = cube.dp_shape
    return (qx / 2.0, qy / 2.0)
