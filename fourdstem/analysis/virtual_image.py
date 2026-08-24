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


def _otsu_threshold(x, nbins=256):
    """Otsu's threshold (numpy only) — the valley between two intensity modes."""
    x = np.asarray(x, float).ravel()
    x = x[np.isfinite(x)]
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return lo
    hist, edges = np.histogram(x, bins=nbins, range=(lo, hi))
    p = hist.astype(float) / max(hist.sum(), 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    w = np.cumsum(p)
    mu = np.cumsum(p * centers)
    muT = mu[-1]
    denom = w * (1.0 - w)
    safe = denom > 1e-12
    sb = np.zeros_like(denom)
    sb[safe] = (muT * w[safe] - mu[safe]) ** 2 / denom[safe]
    return float(centers[int(np.argmax(sb))])


def material_mask(cube, center=None, r_inner=None, r_outer=None, empty_mask=None,
                  thresh_k=3.0, percentile=None, min_frac=0.02):
    """Boolean scan map of positions that contain material (vs vacuum/empty).

    Empty (no-sample) probe positions scatter almost nothing into the structural
    ring, so an annular scattering image is dark there. This thresholds that image
    to keep only material positions, so they can be *excluded* from the analysis
    (radial profiles, reduction, decomposition, fluctuation) instead of diluting
    it with vacuum.

    Threshold, in priority order:
      * ``empty_mask`` given — a scan-shaped bool marking a known empty region:
        threshold = ``mean + thresh_k*std`` of the empty positions' scattering
        (robust, tied to the real vacuum level).
      * ``percentile`` given — that percentile of the scattering image.
      * otherwise — Otsu's threshold (automatic two-mode valley).

    Returns a scan-shaped bool (True = material). ``min_frac`` guards against a
    degenerate mask (if fewer than this fraction survive, keep the top ``min_frac``).
    """
    center = _resolve_center(cube, center)
    dp = cube.dp_shape
    if r_inner is None:
        r_inner = min(dp) / 8.0
    if r_outer is None:
        r_outer = min(dp) / 2.5
    scat = np.asarray(annular_dark_field(cube, center=center,
                                         r_inner=r_inner, r_outer=r_outer), float)
    if empty_mask is not None:
        e = scat[np.asarray(empty_mask, bool)]
        # robust (median + k*MAD) so a few material pixels leaking into the empty
        # region don't inflate the threshold and over-exclude the sample
        med = float(np.median(e))
        mad = 1.4826 * float(np.median(np.abs(e - med)))
        thr = med + thresh_k * (mad if mad > 0 else float(e.std()))
    elif percentile is not None:
        thr = float(np.percentile(scat, percentile))
    else:
        thr = _otsu_threshold(scat)
    mask = scat > thr
    if mask.mean() < min_frac:                       # degenerate -> keep the brightest
        thr = float(np.percentile(scat, 100 * (1 - min_frac)))
        mask = scat > thr
    return mask


def average_pattern(cube, scan_mask):
    """Mean diffraction pattern over the scan positions where ``scan_mask`` is True.

    The workhorse for region- or distance-resolved analysis: pick a set of scan
    positions (a phase region, a distance band from an interface) and average
    their patterns into one high-SNR diffraction pattern to feed to
    :func:`~fourdstem.analysis.rdf.pattern_to_rdf`.
    """
    flat = cube._flat_patterns()
    m = np.asarray(scan_mask, bool).ravel()
    if m.sum() == 0:
        return np.full(cube.dp_shape, np.nan)
    return flat[m].mean(0, dtype=np.float64)


def average_pattern_aligned(cube, scan_mask, target=None, threshold=0.3):
    """Mean pattern over masked positions, each SHIFTED to a common beam center.

    When the direct beam wanders across the scan (imperfect descan / a tilted
    beam over a wide area), a plain :func:`average_pattern` smears the rings
    because it stacks patterns whose centers differ. Here each pattern is first
    integer-shifted so its beam center (center of mass of the bright core above
    ``threshold``) lands on ``target`` (default the detector center), then
    averaged — giving a sharp mean pattern suitable for a fixed-center RDF.
    Loops over the masked positions only, so it stays light for a phase region.
    """
    flat = cube._flat_patterns()
    dp = cube.dp_shape
    H, W = dp
    tx, ty = target if target is not None else (W / 2.0, H / 2.0)
    yy, xx = np.mgrid[0:H, 0:W]
    idx = np.where(np.asarray(scan_mask, bool).ravel())[0]
    if idx.size == 0:
        return np.full(dp, np.nan)
    acc = np.zeros(dp, np.float64)
    for i in idx:
        p = np.asarray(flat[i], np.float64)
        pk = p.max()
        w = np.where(p >= threshold * pk, p, 0.0) if pk > 0 else p
        tot = w.sum()
        if tot > 0:
            cx = (w * xx).sum() / tot
            cy = (w * yy).sum() / tot
        else:
            cx, cy = tx, ty
        acc += np.roll(np.roll(p, int(round(ty - cy)), 0), int(round(tx - cx)), 1)
    return acc / idx.size


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
