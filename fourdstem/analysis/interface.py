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


def _row_signals(work, sign, smooth, broad_win, row_detrend):
    """Per-row horizontal profiles, high-pass detrended and sign-corrected so the
    interface is a positive peak in every row (rows = axis 0, x = axis 1)."""
    Sy, Sx = work.shape
    if row_detrend:
        work = work - np.median(work, axis=1, keepdims=True)
    bw = broad_win or max(11, (Sx // 4) | 1)
    R = np.empty_like(work, float)
    for y in range(Sy):
        row = _smooth1d(work[y], smooth)
        R[y] = sign * (row - _rolling_median(row, bw))
    return R


def _half_max_band(row, pk, floor):
    """Half-maximum crossings of a single-row profile around peak index ``pk``."""
    n = row.size
    half = floor + 0.5 * (row[pk] - floor)
    l = pk
    while l > 0 and row[l - 1] >= half:
        l -= 1
    r = pk
    while r < n - 1 and row[r + 1] >= half:
        r += 1
    return l, r


def localize_interface(cube, center=None, feature="bf", rings=None,
                       bf_radius=None, axis="vertical", row_detrend=True,
                       smooth=3, broad_win=None, line_sign="auto",
                       fit_halfwidth=8, band_sigma=2.0, bulk_gap=3.0,
                       min_width=1.0, per_row=False, search_halfwidth=None,
                       edge_margin=None, prominence_frac=0.35):
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
        Localize the interface with a **horizontal line profile per row**: track
        the peak near the running (connected) center, measure its **half-max
        (FWHM) band**, and take the union of the per-row bands as the interface
        AREA. Rows whose peak is not prominent enough (``prominence_frac``) are
        dropped, so only the CONTINUOUS line contributes. Handles tilt and a
        temperature-varying width; ``width`` is then the median per-row FWHM.
    search_halfwidth : int, optional
        Per-row tracking half-window (px) around the running center.
    edge_margin : int, optional
        Columns to ignore at each x-edge when detecting the line — suppresses
        rolling-median/high-pass boundary spikes that otherwise capture the
        localizer at the scan border. Default ``~Sx/20``.
    prominence_frac : float
        Per-row connectivity gate: keep a row only if its peak prominence is at
        least this fraction of the median row prominence.

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
    if edge_margin is None:
        edge_margin = max(3, Sx // 20)              # ignore boundary artifacts
    m0, m1 = int(edge_margin), int(Sx - edge_margin)

    prof = _line_profile(work, row_detrend, smooth, broad_win)
    if line_sign == "auto":
        core = prof[m0:m1]                          # decide sign away from edges
        sign = -1.0 if abs(core.min()) >= abs(core.max()) else 1.0
    else:
        sign = -1.0 if str(line_sign).lower() in ("dark", "dip", "neg", "-") else 1.0
    sp = sign * prof                                # interface is now a peak
    x0 = m0 + int(np.argmax(sp[m0:m1]))             # global center, edges excluded
    xs = np.arange(Sx, dtype=float)

    g = fit_gaussian_peak(xs, sp, max(m0, x0 - fit_halfwidth),
                          min(m1 - 1, x0 + fit_halfwidth))
    if g["success"] and np.isfinite(g["center"]) and abs(g["center"] - x0) <= fit_halfwidth:
        x_if = float(g["center"])
        sigma = float(max(g["sigma"], 0.5))
    else:
        x_if = float(x0)
        sigma = 1.0
    contrast = float(sp[x0])

    cols = np.arange(Sx)[None, :]

    if per_row:
        # Horizontal per-row line profiles. In each row track the interface peak
        # near the (connected) running center, measure its half-max band, and
        # gate weak rows so only the CONTINUOUS line becomes area.
        R = _row_signals(work, sign, smooth, broad_win, row_detrend)
        hw = int(search_halfwidth or max(4, round(4 * sigma)))
        centers = np.full(Sy, x_if)
        proms = np.zeros(Sy)
        bands = [None] * Sy
        track = x_if
        for y in range(Sy):
            lo = max(m0, int(round(track)) - hw)
            hi = min(m1, int(round(track)) + hw + 1)
            if hi - lo < 3:
                lo, hi = m0, m1
            seg = R[y, lo:hi]
            pk = lo + int(np.argmax(seg))
            floor = float(np.median(R[y, m0:m1]))
            proms[y] = R[y, pk] - floor
            centers[y] = pk
            bands[y] = _half_max_band(R[y], pk, floor)
            track = 0.5 * track + 0.5 * pk           # connectivity: follow the line
        # connectivity gate: keep rows whose peak is prominent vs the median
        ref = np.median(proms[proms > 0]) if np.any(proms > 0) else 0.0
        good = proms >= max(prominence_frac * ref, 1e-9)
        # smooth the kept centers; interpolate the line across gaps for display
        line = centers.astype(float)
        if good.any():
            yy = np.arange(Sy)
            line = np.interp(yy, yy[good], _smooth1d(centers[good].astype(float), 3))
        interface_mask = np.zeros((Sy, Sx), bool)
        fwhms = []
        for y in range(Sy):
            if not good[y]:
                continue
            l, r = bands[y]
            l = max(m0, min(l, Sx - 1)); r = max(l, min(r, Sx - 1))
            interface_mask[y, l:r + 1] = True
            fwhms.append(r - l + 1)
        width = float(np.median(fwhms)) if fwhms else float(max(min_width, 2.3548 * sigma))
        dmap = np.abs(cols - line[:, None])
        sig_eff = max(width / 2.3548, sigma)
        bulk_mask = (dmap >= max(bulk_gap * sig_eff, width / 2 + 2.0)) & ~interface_mask
    else:
        width = float(max(min_width, 2.3548 * sigma))
        line = np.full(Sy, x_if, float)
        dmap = np.abs(cols - line[:, None])
        half_w = max(min_width, band_sigma * sigma)
        interface_mask = dmap <= half_w
        bulk_mask = dmap >= max(bulk_gap * sigma, half_w + 1.0)

    if bulk_mask.sum() < 5:                          # keep a real bulk set
        bulk_mask = (dmap >= np.percentile(dmap, 60)) & ~interface_mask

    if axis == "horizontal":                         # undo transpose for outputs
        dmap = dmap.T
        interface_mask, bulk_mask = interface_mask.T, bulk_mask.T

    return dict(s=s, prof=sp, x_if=x_if, line=line, sigma=sigma,
                width=width, contrast=contrast, dmap=dmap,
                interface_mask=interface_mask, bulk_mask=bulk_mask,
                interface_area=int(interface_mask.sum()), center=center)
