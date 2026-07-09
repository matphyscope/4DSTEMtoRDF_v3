"""Interface localization — classify scan positions into interface vs bulk from
the diffraction *structure* (a virtual image), WITHOUT any clustering or matrix
factorization (no k-means / NMF / PCA).

A buried amorphous interface is a thin, *vertical* band: the same scan column x
across every row. In a bright-field image it reads as a dark vertical line; in a
brightness-cancelled structural (ring-DF/total) map it reads as a contrast line.
Real in-situ scans bury that line under strong *horizontal* scan-line striping
(a per-row DC offset) and a slow left-to-right gradient. This module removes both
— per-row median subtraction kills the horizontal stripes, a rolling-median
high-pass removes the broad gradient — then collapses to a 1-D column profile and
**Gaussian-fits the sharp line** for a sub-pixel center and a real FWHM width. The
result is interface / bulk masks, a distance-from-interface map, the fitted width
and contrast, and the interface area — every downstream quantity (per-phase RDF,
interface area, distance-resolved SRO/MRO) follows the localized band, not an
intensity threshold that would bleed across the boundary.
"""
from __future__ import annotations

import numpy as np

__all__ = ["localize_interface"]


def _smooth1d(y, w):
    w = int(w or 0)
    if w > 1:
        return np.convolve(y, np.ones(w) / w, mode="same")
    return y


def _rolling_median(y, win):
    """Broad background of a 1-D profile (for high-pass detrend)."""
    win = int(win)
    if win < 3:
        return np.zeros_like(y, float)
    try:
        from scipy.ndimage import median_filter
        return median_filter(np.asarray(y, float), size=win, mode="reflect")
    except Exception:
        # coarse fallback: moving average
        k = np.ones(win) / win
        return np.convolve(np.asarray(y, float), k, mode="same")


def _feature_map(cube, feature, center, rings, bf_radius):
    from .virtual_image import (bright_field, annular_dark_field,
                                structural_map)
    if isinstance(feature, np.ndarray):
        return np.asarray(feature, float)
    if feature == "bf":
        r = bf_radius if bf_radius is not None else max(3.0, min(cube.dp_shape) / 12.0)
        return np.asarray(bright_field(cube, center=center, radius=r), float)
    if feature == "adf":
        return np.asarray(annular_dark_field(cube, center=center), float)
    if feature == "structural":
        m = min(cube.dp_shape)
        if rings is None:
            rings = [(m / 5.0, m / 3.0)]
        maps = [np.asarray(structural_map(cube, center=center,
                                          r_inner=ri, r_outer=ro), float)
                for ri, ro in rings]
        return np.mean(maps, axis=0)
    raise ValueError(f"unknown feature {feature!r}")


def _line_profile(work, row_detrend, smooth, broad_win):
    """Collapse a 2-D feature map (line varies along axis 1) to a detrended 1-D
    column profile in which the vertical interface is a sharp *positive-or-
    negative* spike, free of horizontal stripes and the broad x-gradient."""
    Sy, Sx = work.shape
    if row_detrend:
        work = work - np.median(work, axis=1, keepdims=True)   # kill h-stripes
    p = _smooth1d(work.mean(0), smooth)                        # column profile
    if broad_win is None:
        broad_win = max(11, (Sx // 4) | 1)
    return p - _rolling_median(p, broad_win)                   # high-pass


def localize_interface(cube, center=None, feature="bf", rings=None,
                       bf_radius=None, axis="vertical", row_detrend=True,
                       smooth=3, broad_win=None, line_sign="auto",
                       fit_halfwidth=8, band_sigma=2.0, bulk_gap=3.0,
                       min_width=1.0, per_row=False, search_halfwidth=None):
    """Locate a thin interface line and split the scan into interface / bulk — no ML.

    Parameters
    ----------
    cube : DataCube
    center : (cx, cy), optional
        Diffraction center; auto-found if omitted.
    feature : {"bf", "structural", "adf"} or 2-D array
        Virtual image the interface lives in. ``"bf"`` (default) = bright-field,
        where a real interface is a dark vertical line — most robust on real
        data. ``"structural"`` = ring-DF/total (pass ``rings``). Or pass your own
        scan-shaped map.
    rings : list of (r_inner, r_outer)
        Structural bands (px) for ``feature="structural"``.
    bf_radius : float, optional
        Bright-field disk radius (px). Default ``min(Qy,Qx)/12``.
    axis : {"vertical", "horizontal"}
        Interface line orientation.
    row_detrend : bool
        Subtract each row's median first — removes horizontal scan-line striping
        (a per-row DC offset) so the vertical line dominates. Key on real data.
    smooth : int
        Boxcar width (px) for profile smoothing.
    broad_win : int, optional
        Rolling-median window (px) for the high-pass that removes the slow
        left-right gradient. Default ``~Sx/4``.
    line_sign : {"auto", "dark", "bright"}
        Whether the interface is a dip (``"dark"``, e.g. BF) or a peak
        (``"bright"``). ``"auto"`` picks the larger deviation.
    fit_halfwidth : int
        Half-window (px) for the Gaussian line fit around the detected spike.
    band_sigma : float
        Interface band half-width = ``band_sigma * sigma`` of the fitted line.
    bulk_gap : float
        Bulk = positions at distance ``>= bulk_gap * sigma`` from the line.
    min_width : float
        Floor on the reported width / band half-width (px).
    per_row : bool
        Localize the line per row (tilt-robust) within ``search_halfwidth`` of
        the global center.
    search_halfwidth : int, optional
        Per-row search half-window (px).

    Returns
    -------
    dict with keys ``s`` (feature map), ``prof`` (detrended 1-D profile, line
    as a positive peak), ``x_if`` (sub-pixel center), ``line`` (per-row centers),
    ``sigma``, ``width`` (FWHM px), ``contrast``, ``dmap`` (distance map),
    ``interface_mask`` / ``bulk_mask``, ``interface_area``, ``center``.
    """
    from .virtual_image import _resolve_center
    from .peaks import fit_gaussian_peak

    center = _resolve_center(cube, center)
    s = _feature_map(cube, feature, center, rings, bf_radius)
    if s.ndim != 2:
        raise ValueError("localize_interface needs a 2-D scan grid")

    work = s.T if axis == "horizontal" else s      # line now varies along axis 1
    Sy, Sx = work.shape

    prof = _line_profile(work, row_detrend, smooth, broad_win)
    if line_sign == "auto":
        sign = -1.0 if abs(prof.min()) >= abs(prof.max()) else 1.0
    else:
        sign = -1.0 if str(line_sign).lower() in ("dark", "dip", "neg", "-") else 1.0
    sp = sign * prof                                # interface is now a peak
    x0 = int(np.argmax(sp))
    xs = np.arange(Sx, dtype=float)

    g = fit_gaussian_peak(xs, sp, max(0, x0 - fit_halfwidth),
                          min(Sx - 1, x0 + fit_halfwidth))
    if g["success"] and np.isfinite(g["center"]) and abs(g["center"] - x0) <= fit_halfwidth:
        x_if = float(g["center"])
        sigma = float(max(g["sigma"], 0.5))
    else:
        x_if = float(x0)
        sigma = 1.0
    width = float(max(min_width, 2.3548 * sigma))
    contrast = float(sp[x0])

    if per_row:
        hw = int(search_halfwidth or max(3, round(3 * sigma)))
        line = np.empty(Sy, float)
        base = work - np.median(work, axis=1, keepdims=True) if row_detrend else work
        for y in range(Sy):
            row = sign * (_smooth1d(base[y], smooth)
                          - _rolling_median(_smooth1d(base[y], smooth),
                                            broad_win or max(11, (Sx // 4) | 1)))
            lo, hi = max(0, int(x_if) - hw), min(Sx, int(x_if) + hw + 1)
            line[y] = lo + int(np.argmax(row[lo:hi]))
        line = _smooth1d(line, 3)
    else:
        line = np.full(Sy, x_if, float)

    cols = np.arange(Sx)[None, :]
    dmap = np.abs(cols - line[:, None])

    half_w = max(min_width, band_sigma * sigma)
    interface_mask = dmap <= half_w
    bulk_mask = dmap >= max(bulk_gap * sigma, half_w + 1.0)
    if bulk_mask.sum() < 5:                          # keep a real bulk set
        bulk_mask = dmap >= np.percentile(dmap, 60)

    if axis == "horizontal":                         # undo transpose for outputs
        dmap = dmap.T
        interface_mask, bulk_mask = interface_mask.T, bulk_mask.T

    return dict(s=s, prof=sp, x_if=x_if, line=line, sigma=sigma,
                width=width, contrast=contrast, dmap=dmap,
                interface_mask=interface_mask, bulk_mask=bulk_mask,
                interface_area=int(interface_mask.sum()), center=center)
