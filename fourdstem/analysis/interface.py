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


def _robust_polyfit(y, x, w, order, n_iter=2):
    """Weighted polynomial fit x(y) with iterative outlier down-weighting — gives
    a smooth, near-straight interface line that ignores noisy wandering rows."""
    order = int(max(0, order))
    yy = np.asarray(y, float)
    xx = np.asarray(x, float)
    ww = np.clip(np.asarray(w, float), 0, None) + 1e-9
    coef = np.polyfit(yy, xx, order, w=ww)
    for _ in range(n_iter):
        pred = np.polyval(coef, yy)
        res = np.abs(xx - pred)
        mad = np.median(res) + 1e-9
        keep = res <= 3.0 * mad
        if keep.sum() <= order + 1:
            break
        coef = np.polyfit(yy[keep], xx[keep], order, w=ww[keep])
    return coef


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
                       edge_margin=None, prominence_frac=0.35,
                       line_order=1, min_snr=4.0, min_good_frac=0.5,
                       n_seed=10, corridor=None, coherence_max=None):
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
    line_order : int
        Degree of the robust polynomial fit through the per-row centers.
        ``0`` = perfectly vertical (constant x), ``1`` = straight with tilt
        (default; straightens a wavy track), ``2`` = gentle curve.
    min_snr : float
        Presence threshold: the global peak prominence must exceed ``min_snr``
        times the profile's robust noise or the interface is declared ABSENT
        (``present=False``, area/width = 0). Distinguishes a real line from a
        homogenized scan (e.g. above the interface-disappearance temperature).
    min_good_frac : float
        Presence also requires at least this fraction of rows to carry a
        prominent, connected peak.
    n_seed : int
        Two-pass anchor: the ``n_seed`` most prominent ("most confident") rows
        seed a robust line; every row is then searched only within a corridor
        around that line, so noisy rows cannot wander off.
    corridor : int, optional
        Half-width (px) of the vertical search corridor around the anchor. The
        fitted line is clipped to this corridor, so a residual tilt can never
        become a runaway diagonal. Increase it for a genuinely tilted interface.
    coherence_max : float, optional
        Presence gate: the confident seed rows must AGREE on x to within this
        spread (px). Scattered seed peaks (no coherent line — e.g. a homogenized
        scan that produced a spurious diagonal) → ABSENT. Default ``~3*sigma``.

    Notes
    -----
    Also returns ``present`` (bool), ``snr``, and ``good_frac`` so a temperature
    series can drop the interface where it has homogenized.

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

    # Global presence SNR: peak prominence vs the profile's robust noise away
    # from the line. A homogenized (no-interface) scan has a low SNR.
    away = np.abs(np.arange(m0, m1) - x0) > max(3, 2 * sigma)
    core = sp[m0:m1]
    noise = 1.4826 * np.median(np.abs(core[away] - np.median(core[away]))) if away.any() else core.std()
    snr = float(contrast / noise) if noise > 0 else float("inf")

    cols = np.arange(Sx)[None, :]

    if per_row:
        # Two-pass horizontal per-row localization:
        #  (1) find each row's best peak; take the N most prominent ("confident")
        #      rows and fit a robust near-straight line through them -> a corridor.
        #  (2) search every row ONLY inside that corridor, so noisy rows cannot
        #      wander off. Then gate weak rows and take the half-max (FWHM) band.
        R = _row_signals(work, sign, smooth, broad_win, row_detrend)
        yy = np.arange(Sy)
        floors = np.median(R[:, m0:m1], axis=1)

        corr = int(corridor or max(4, round(3 * sigma)))
        cmax = float(coherence_max if coherence_max is not None else max(4.0, 3 * sigma))

        # pass 1: unconstrained peak + prominence per row (edges excluded)
        pk1 = m0 + np.argmax(R[:, m0:m1], axis=1)
        prom1 = R[yy, pk1] - floors
        n_seed = int(min(max(3, n_seed), Sy))
        seed = np.argsort(prom1)[-n_seed:]            # most confident rows
        seed = seed[prom1[seed] > 0]
        # Anchor the corridor VERTICALLY on the position the confident rows agree
        # on (robust median), reconciled with the global column-profile center —
        # NOT a tilted fit, which overfits a diagonal when the line is weak.
        seed_center = float(np.median(pk1[seed])) if seed.size else x_if
        if abs(seed_center - x_if) > corr:
            seed_center = x_if
        # coherence: do the confident rows actually AGREE on an x? (scattered = no line)
        seed_spread = (float(np.median(np.abs(pk1[seed] - np.median(pk1[seed]))))
                       if seed.size else float("inf"))
        seed_line = np.full(Sy, seed_center)

        # pass 2: constrained peak inside the vertical corridor around the anchor
        centers = np.empty(Sy)
        proms = np.empty(Sy)
        for y in range(Sy):
            lo = max(m0, int(round(seed_line[y])) - corr)
            hi = min(m1, int(round(seed_line[y])) + corr + 1)
            if hi - lo < 3:
                lo, hi = m0, m1
            pk = lo + int(np.argmax(R[y, lo:hi]))
            centers[y] = pk
            proms[y] = R[y, pk] - floors[y]
        ref = np.median(proms[proms > 0]) if np.any(proms > 0) else 0.0
        good = proms >= max(prominence_frac * ref, 1e-9)
        good_frac = float(good.mean())

        # robust low-order fit through the constrained rows, then CLIP to the
        # corridor so a residual tilt can never become a runaway diagonal.
        if good.sum() > line_order + 1:
            coef = _robust_polyfit(yy[good], centers[good], proms[good], line_order)
            line = np.polyval(coef, yy)
        else:
            line = seed_line.astype(float)
        line = np.clip(line, seed_center - corr, seed_center + corr)
        line = np.clip(line, m0, m1 - 1)

        present = ((snr >= min_snr) and (good_frac >= min_good_frac)
                   and (seed_spread <= cmax))

        # FWHM band measured on the fitted line (so the ribbon is smooth)
        interface_mask = np.zeros((Sy, Sx), bool)
        fwhms = []
        if present:
            for y in range(Sy):
                if not good[y]:
                    continue
                pk = int(round(line[y]))
                floor = float(np.median(R[y, m0:m1]))
                l, r = _half_max_band(R[y], pk, floor)
                l = max(m0, min(l, Sx - 1)); r = max(l, min(r, Sx - 1))
                interface_mask[y, l:r + 1] = True
                fwhms.append(r - l + 1)
        width = float(np.median(fwhms)) if fwhms else 0.0
        area = int(interface_mask.sum())
        if not present:                              # keep a 1-px line for RDF, area=0
            interface_mask = np.abs(cols - line[:, None]) <= 0.5
        dmap = np.abs(cols - line[:, None])
        sig_eff = max(width / 2.3548, sigma)
        bulk_mask = (dmap >= max(bulk_gap * sig_eff, width / 2 + 2.0)) & ~interface_mask
    else:
        present = snr >= min_snr
        good_frac = 1.0
        width = float(max(min_width, 2.3548 * sigma))
        line = np.full(Sy, x_if, float)
        dmap = np.abs(cols - line[:, None])
        half_w = max(min_width, band_sigma * sigma)
        interface_mask = dmap <= half_w
        bulk_mask = dmap >= max(bulk_gap * sigma, half_w + 1.0)
        area = int(interface_mask.sum()) if present else 0

    if bulk_mask.sum() < 5:                          # keep a real bulk set
        bulk_mask = (dmap >= np.percentile(dmap, 60)) & ~interface_mask

    if axis == "horizontal":                         # undo transpose for outputs
        dmap = dmap.T
        interface_mask, bulk_mask = interface_mask.T, bulk_mask.T

    return dict(s=s, prof=sp, x_if=x_if, line=line, sigma=sigma,
                width=width, contrast=contrast, dmap=dmap,
                interface_mask=interface_mask, bulk_mask=bulk_mask,
                interface_area=int(area), present=bool(present),
                snr=snr, good_frac=float(good_frac), center=center)
