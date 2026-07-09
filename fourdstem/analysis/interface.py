"""Interface localization — classify scan positions into interface vs bulk from
the diffraction *structure* (a q-contrast virtual image), WITHOUT any clustering
or matrix factorization (no k-means / NMF / PCA).

A buried amorphous interface is a thin spatial band whose short/medium-range
order differs from the bulk. In a brightness-cancelled structural map
(``structural_map`` = ring-DF / total) it shows up as a bright line, even when it
is far too thin a minority for pixel clustering to isolate. This module finds
that line geometrically and returns interface / bulk masks plus a
distance-from-interface map, so every downstream quantity (per-phase RDF,
interface area, distance-resolved SRO/MRO) follows the localized band instead of
an intensity threshold that would bleed across the boundary.
"""
from __future__ import annotations

import numpy as np

__all__ = ["localize_interface"]


def _smooth1d(y, w):
    if w and w > 1:
        k = np.ones(int(w)) / float(int(w))
        return np.convolve(y, k, mode="same")
    return y


def localize_interface(cube, center=None, rings=None, detrend=True,
                       detrend_sigma=6.0, axis="vertical", per_row=False,
                       smooth=3, search_halfwidth=None, bulk_gap=2.0):
    """Locate an interface band from the structural (q-contrast) map — no ML.

    Parameters
    ----------
    cube : DataCube
    center : (cx, cy), optional
        Diffraction center; auto-found if omitted.
    rings : list of (r_inner, r_outer), optional
        Structural bands (px) to average for the contrast map. Default is the
        amorphous-ring band used by :func:`structural_map`. Pick the band that
        best separates the interface (see the q-contrast scan, nb3 §1c).
    detrend : bool
        Subtract a smooth Gaussian background (``detrend_sigma`` px) from the
        structural map first, removing the slow dose/thickness gradient so a
        sharp interface line is not swamped by it.
    axis : {"vertical", "horizontal"}
        Orientation of the interface line. ``"vertical"`` = a line at some scan
        column x (profile taken across columns); ``"horizontal"`` transposes.
    per_row : bool
        If True, localize the line independently on each row (each column, for a
        horizontal interface) within ``search_halfwidth`` of the global peak —
        robust to a *tilted* interface. If False, one straight line at the global
        profile peak.
    smooth : int
        Boxcar width for profile smoothing (px).
    search_halfwidth : int, optional
        Per-row search half-window around the global peak (px). Default scales
        with the measured width.
    bulk_gap : float
        Bulk positions are those at distance ``>= bulk_gap * width`` from the
        line (well clear of the band).

    Returns
    -------
    dict with keys:
        ``s`` structural map, ``resid`` detrended map, ``prof`` 1-D profile,
        ``x_if`` global line index, ``line`` per-row line indices,
        ``width`` band FWHM (px), ``contrast`` peak-minus-baseline,
        ``dmap`` distance-from-interface map (scan-shaped, px),
        ``interface_mask`` / ``bulk_mask`` (scan-shaped bool),
        ``interface_area`` (# interface positions), ``center``.
    """
    from .virtual_image import structural_map, _resolve_center

    center = _resolve_center(cube, center)
    dp = cube.dp_shape
    m = min(dp)
    if rings is None:
        rings = [(m / 5.0, m / 3.0)]
    maps = [np.asarray(structural_map(cube, center=center,
                                      r_inner=ri, r_outer=ro), float)
            for ri, ro in rings]
    s = np.mean(maps, axis=0)
    if s.ndim != 2:
        raise ValueError("localize_interface needs a 2-D scan grid")

    # transpose so the interface line is always "vertical" (varies along axis 1)
    work = s.T if axis == "horizontal" else s
    Sy, Sx = work.shape

    resid = work
    if detrend:
        try:
            from scipy.ndimage import gaussian_filter
            resid = work - gaussian_filter(work, detrend_sigma)
        except Exception:
            resid = work - np.median(work)

    prof = _smooth1d(resid.mean(0), smooth)
    base = float(np.median(prof))
    x0 = int(np.argmax(prof))
    peak = float(prof.max())
    contrast = peak - base
    if contrast > 1e-6:
        above = np.where(prof >= base + 0.5 * contrast)[0]
        width = float(above.max() - above.min() + 1) if above.size else 1.0
    else:
        width = 0.0

    if per_row:
        hw = int(search_halfwidth or max(3, round(2 * max(width, 2.0))))
        line = np.empty(Sy, int)
        for y in range(Sy):
            row = _smooth1d(resid[y], smooth)
            lo, hi = max(0, x0 - hw), min(Sx, x0 + hw + 1)
            line[y] = lo + int(np.argmax(row[lo:hi]))
    else:
        line = np.full(Sy, x0, int)

    cols = np.arange(Sx)[None, :]
    dmap = np.abs(cols - line[:, None]).astype(float)

    half_w = max(1.0, width / 2.0)
    interface_mask = dmap <= half_w
    bulk_mask = dmap >= bulk_gap * max(width, 1.0)
    if bulk_mask.sum() < 5:                      # degenerate: keep a real bulk set
        bulk_mask = dmap >= np.percentile(dmap, 60)

    if axis == "horizontal":                     # undo the transpose for outputs
        s, resid, dmap = s, resid.T, dmap.T
        interface_mask, bulk_mask = interface_mask.T, bulk_mask.T

    return dict(s=s, resid=resid, prof=prof, x_if=x0, line=line,
                width=width, contrast=float(contrast), dmap=dmap,
                interface_mask=interface_mask, bulk_mask=bulk_mask,
                interface_area=int(interface_mask.sum()), center=center)
