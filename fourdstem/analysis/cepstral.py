"""Exit-wave power cepstrum (EWPC) and fluctuation cepstral STEM (FC-STEM).

Implements the cepstral 4D-STEM analysis of Pidaparthy, Ni, Hou, Abraham & Zuo,
*Ultramicroscopy* 248 (2023) 113718 — well suited to WEAK-signal, mixed-phase
amorphous materials (e.g. lithiated silicon battery anodes) where the usual
structure-factor reduction fails.

The exit-wave power cepstrum of a diffraction pattern ``I(k)`` is

    EWPC(r) = | F^{-1} { log( I(k) ) } |                        (paper Eq. 2)

The log compresses the huge dynamic range so weak scattering survives, and no
background subtraction / atomic-scattering-factor reduction is needed. The
quefrency ``r`` is a real-space distance (Å): peaks fall at inter-atomic
vectors, so the azimuthal average of EWPC is an RDF-like profile.

Phase mapping uses the fluctuation (normalized variance) of the cepstrum over an
annular quefrency band ``S = {r | r_in < r < r_out}`` at each probe position:

    F(Rp) = <Cp^2>_{r in S} / <Cp>^2_{r in S} - 1               (paper Eq. 3)

Different bands (distance ranges) highlight different ordered/disordered phases.

Quefrency calibration: a diffraction pattern sampled at ``q_per_px`` (Å^-1/px)
over ``N`` pixels has cepstral pixel size ``dr = 1 / (N * q_per_px)`` Å.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "ewpc_pattern", "quefrency_per_px", "cepstral_radial_profile",
    "fluctuation_image", "fluctuation_multiband", "fluctuation_profile", "cepstral_peakiness_image", "cepstral_discreteness_image", "cepstral_lattice_gvectors", "spot_lattice", "cepstral_periodicity", "ewpc_mean", "ewpc_profiles",
]


def ewpc_pattern(pattern, offset=1.0, subtract_mean=True, window=True, pad=None,
                 highpass=None):
    """Exit-wave power cepstrum of one diffraction pattern (fftshifted, r=0 center).

    ``EWPC = |F^{-1}{ log(I + offset) }|``. ``offset`` keeps the log finite over
    zeros/low counts. ``window`` applies a 2-D Hann window before the transform to
    suppress the cross-shaped artifact from the pattern edges. ``subtract_mean``
    removes the mean of log I so the r=0 (DC) spike doesn't dominate — it only
    affects the central pixel, never the analysis band.

    ``pad`` (int) zero-pads the windowed log-pattern to ``pad × pad`` before the
    inverse transform, interpolating the cepstrum to a finer quefrency step
    ``dr = 1/(pad · q_per_px)`` (the paper pads to 1024). Needed when the raw
    detector is small enough that ``dr`` is coarser than the analysis band width.
    The returned pattern is then ``pad × pad``; pass the same ``pad`` as ``n`` to
    :func:`quefrency_per_px`.

    ``highpass`` (Gaussian sigma in detector px) subtracts a blurred copy of
    ``log I`` first, removing the smooth central-beam + halo envelope. Without it
    that envelope dominates the cepstrum as a big central blob whose flank ripples
    swamp the true lattice peaks; with it the discrete lattice peaks stand out.
    Peak positions (hence lattice vectors) are unchanged. Used by
    :func:`cepstral_lattice_gvectors`.
    """
    p = np.asarray(pattern, float)
    logI = np.log(np.maximum(p, 0.0) + offset)
    if subtract_mean:
        logI = logI - logI.mean()
    if highpass:
        from scipy.ndimage import gaussian_filter
        logI = logI - gaussian_filter(logI, float(highpass))
    if window:
        h0 = np.hanning(logI.shape[0])
        h1 = np.hanning(logI.shape[1])
        logI = logI * np.outer(h0, h1)
    if pad:
        pad = int(pad)
        if pad < max(logI.shape):
            raise ValueError(f"pad={pad} smaller than pattern {logI.shape}")
        buf = np.zeros((pad, pad), float)
        y0 = (pad - logI.shape[0]) // 2
        x0 = (pad - logI.shape[1]) // 2
        buf[y0:y0 + logI.shape[0], x0:x0 + logI.shape[1]] = logI
        logI = buf
    cep = np.fft.fftshift(np.fft.ifft2(logI))
    return np.abs(cep)


def quefrency_per_px(n, q_per_px):
    """Cepstral pixel size in Å: ``dr = 1/(N*q_per_px)`` for an N-pixel axis."""
    return 1.0 / (float(n) * float(q_per_px))


def _radius_px(shape):
    n0, n1 = shape
    yy, xx = np.mgrid[0:n0, 0:n1]
    return np.hypot(yy - n0 // 2, xx - n1 // 2)


def cepstral_radial_profile(cep, q_per_px, n_bins=None, r_min=0.3):
    """Azimuthal average of a cepstral pattern -> ``(r_A, profile)``.

    ``r_A`` is quefrency in Å. ``r_min`` (Å) drops the central DC region.
    """
    cep = np.asarray(cep, float)
    n = cep.shape[0]
    dr = quefrency_per_px(n, q_per_px)
    rpx = _radius_px(cep.shape)
    r_ang = rpx * dr
    rmax = (min(cep.shape) // 2) * dr
    if n_bins is None:
        n_bins = min(cep.shape) // 2
    edges = np.linspace(0, rmax, n_bins + 1)
    idx = np.clip(np.digitize(r_ang.ravel(), edges) - 1, 0, n_bins - 1)
    flat = cep.ravel()
    prof = np.zeros(n_bins)
    cnt = np.zeros(n_bins)
    np.add.at(prof, idx, flat)
    np.add.at(cnt, idx, 1.0)
    prof = prof / np.where(cnt > 0, cnt, 1.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    keep = centers >= r_min
    return centers[keep], prof[keep]


def _annulus_mask(shape, q_per_px, r_in, r_out):
    dr = quefrency_per_px(shape[0], q_per_px)
    r_ang = _radius_px(shape) * dr
    return (r_ang >= r_in) & (r_ang < r_out)


def fluctuation_image(cube, r_in, r_out, q_per_px, offset=1.0, window=True,
                      mask=None, pad=None, n_jobs=1, progress=False):
    """FC-STEM fluctuation image ``F(Rp)`` over the quefrency band ``[r_in, r_out]`` Å.

    For each probe position, computes the EWPC and the normalized variance of the
    cepstral values whose quefrency radius lies in the band (paper Eq. 3):
    ``F = <Cp^2>/<Cp>^2 - 1``. Bright = large fluctuation (more ordered speckle in
    that distance range). Returns a scan-shaped map. ``mask`` optionally restricts
    the annulus further (bool, cepstrum-shaped). ``pad`` zero-pads each pattern to
    ``pad × pad`` before the EWPC so the quefrency step is fine enough to resolve a
    narrow band (see :func:`ewpc_pattern`). Memory-light (one cepstrum at a time);
    ``n_jobs`` fans positions across cores.
    """
    from ..utils.parallel import parallel_map

    flat = cube._flat_patterns() if hasattr(cube, "_flat_patterns") else \
        np.asarray(cube).reshape(-1, *np.asarray(cube).shape[-2:])
    dp = flat.shape[1:]
    cshape = (int(pad), int(pad)) if pad else dp
    ann = _annulus_mask(cshape, q_per_px, r_in, r_out)
    if mask is not None:
        ann = ann & np.asarray(mask, bool)
    if ann.sum() < 3:
        raise ValueError(
            f"annulus [{r_in},{r_out}] Å selects {int(ann.sum())} cepstral pixels; "
            "widen the band, or pass pad= to refine dr (currently "
            f"{quefrency_per_px(cshape[0], q_per_px):.4g} Å/px).")

    def one(i):
        cep = ewpc_pattern(flat[i], offset=offset, subtract_mean=True,
                           window=window, pad=pad)
        v = cep[ann]
        m = v.mean()
        return float((v * v).mean() / (m * m) - 1.0) if m > 0 else 0.0

    vals = parallel_map(one, range(flat.shape[0]), n_jobs=n_jobs,
                        progress=progress, desc="FC-STEM")
    vals = np.asarray(vals, float)
    scan = cube.scan_shape if hasattr(cube, "scan_shape") else None
    return vals.reshape(scan) if scan and len(scan) == 2 else vals


def fluctuation_multiband(cube, bands, q_per_px, offset=1.0, window=True,
                          pad=None, n_jobs=1, progress=False):
    """FC-STEM fluctuation maps for several quefrency bands at once.

    ``bands`` is a list of ``(r_in, r_out)`` in Å. Computes the EWPC once per
    probe position and evaluates the normalized variance (Eq. 3) for every band —
    far cheaper than calling :func:`fluctuation_image` once per band. ``pad``
    zero-pads each pattern to ``pad × pad`` to refine the quefrency step (see
    :func:`ewpc_pattern`). Returns a list of scan-shaped maps, one per band.
    """
    from ..utils.parallel import parallel_map

    flat = cube._flat_patterns() if hasattr(cube, "_flat_patterns") else \
        np.asarray(cube).reshape(-1, *np.asarray(cube).shape[-2:])
    dp = flat.shape[1:]
    cshape = (int(pad), int(pad)) if pad else dp
    masks = [_annulus_mask(cshape, q_per_px, ri, ro) for ri, ro in bands]
    for (ri, ro), ann in zip(bands, masks):
        if ann.sum() < 3:
            raise ValueError(f"band [{ri},{ro}] Å selects {int(ann.sum())} pixels; "
                             f"pass pad= to refine dr (currently "
                             f"{quefrency_per_px(cshape[0], q_per_px):.4g} Å/px)")

    def one(i):
        cep = ewpc_pattern(flat[i], offset=offset, window=window, pad=pad)
        out = []
        for ann in masks:
            v = cep[ann]; m = v.mean()
            out.append(float((v * v).mean() / (m * m) - 1.0) if m > 0 else 0.0)
        return out

    res = np.asarray(parallel_map(one, range(flat.shape[0]), n_jobs=n_jobs,
                                  progress=progress, desc="FC-STEM bands"))
    scan = cube.scan_shape if hasattr(cube, "scan_shape") else None
    return [(res[:, j].reshape(scan) if scan and len(scan) == 2 else res[:, j])
            for j in range(len(bands))]


def fluctuation_profile(cube, q_per_px, r_min, r_max, width=0.2, step=None,
                        mask=None, offset=1.0, window=True, reducer="mean",
                        pad=None, n_jobs=1, progress=False):
    """FC-STEM fluctuation profile ``F(r)`` vs quefrency (paper Fig. 8).

    Sweeps a narrow quefrency window of fixed ``width`` Å (the paper uses
    ``r_out - r_in = 0.02 nm = 0.2 Å``) across ``[r_min, r_max]``. For each window
    it evaluates the per-position fluctuation ``F(Rp)`` (Eq. 3) and reduces it over
    probe positions (mean by default), giving a single number per quefrency. Peaks
    in ``F(r)`` mark inter-atomic distances at which the phases fluctuate distinctly
    — the paper uses them to pick the annular bands ``[r_in, r_out]`` that separate
    phases. ``mask`` (scan-shaped bool) restricts the probe-position reduction to a
    region (e.g. the material).

    ``reducer`` sets how F is reduced over probe positions at each quefrency:
    ``"mean"`` (paper Fig. 8; best when phases occupy large areas), ``"median"``,
    ``"max"``, or a percentile in ``(0, 100]`` as a float or ``"p95"`` string. A high
    percentile / max surfaces a *minority* phase (e.g. a sparse crystalline grain)
    whose signature the mean would wash out. Returns ``(r_centers, F)``.
    """
    if step is None:
        step = width / 2.0
    centers = np.arange(r_min + width / 2.0, r_max - width / 2.0 + 1e-9, step)
    if centers.size == 0:
        raise ValueError(f"[{r_min},{r_max}] Å too narrow for window width {width} Å")
    bands = [(float(c - width / 2.0), float(c + width / 2.0)) for c in centers]
    maps = fluctuation_multiband(cube, bands, q_per_px, offset=offset,
                                 window=window, pad=pad, n_jobs=n_jobs, progress=progress)
    sm = None if mask is None else np.asarray(mask, bool)

    pctl = None
    if isinstance(reducer, (int, float)) and not isinstance(reducer, bool):
        pctl = float(reducer)
    elif isinstance(reducer, str) and reducer.startswith("p") and reducer[1:].replace(".", "", 1).isdigit():
        pctl = float(reducer[1:])
    if pctl is not None and not (0.0 < pctl <= 100.0):
        raise ValueError(f"percentile reducer must be in (0,100], got {pctl}")

    prof = []
    for m in maps:
        v = m[sm] if sm is not None else np.asarray(m).ravel()
        if pctl is not None:
            prof.append(float(np.percentile(v, pctl)))
        elif reducer == "median":
            prof.append(float(np.median(v)))
        elif reducer == "max":
            prof.append(float(np.max(v)))
        else:
            prof.append(float(np.mean(v)))
    return centers, np.asarray(prof, float)


def cepstral_discreteness_image(cube, r_in, r_out, q_per_px, pad=None, offset=1.0,
                                window=True, smooth=5, mask=None, n_jobs=1,
                                progress=False):
    """Per-pixel crystalline/amorphous map from radial-cepstral discreteness.

    For each probe position: EWPC -> azimuthal (radial) profile over the quefrency
    band ``[r_in, r_out]`` Å -> detrend by a running-median baseline of width
    ``smooth`` bins -> the fraction of profile power sitting in the *positive*
    residual (sharp, discrete peaks) above that smooth baseline. A crystal has
    discrete inter-atomic distances, so its radial cepstrum shows sharp peaks
    (high discreteness); an amorphous phase has a continuous distance distribution,
    so its radial cepstrum is smooth (low discreteness). ``mask`` (scan-shaped
    bool) restricts computation to a region; skipped positions return 0. ``pad``
    refines the quefrency step (see :func:`ewpc_pattern`). Returns a scan-shaped map.
    """
    from ..utils.parallel import parallel_map
    from scipy.ndimage import median_filter

    flat = cube._flat_patterns() if hasattr(cube, "_flat_patterns") else \
        np.asarray(cube).reshape(-1, *np.asarray(cube).shape[-2:])
    scan = cube.scan_shape if hasattr(cube, "scan_shape") else None
    sm = None if mask is None else np.asarray(mask, bool).ravel()
    w = max(3, int(smooth))

    def one(i):
        if sm is not None and not sm[i]:
            return 0.0
        cep = ewpc_pattern(flat[i], offset=offset, window=window, pad=pad)
        r, prof = cepstral_radial_profile(cep, q_per_px, r_min=r_in)
        band = (r >= r_in) & (r <= r_out)
        p = prof[band]
        if p.size < 3 or p.max() <= 0:
            return 0.0
        base = median_filter(p, size=min(w, p.size if p.size % 2 else p.size - 1))
        resid = p - base
        pos = float(resid[resid > 0].sum())
        return pos / (float(base.sum()) + 1e-12)

    vals = np.asarray(parallel_map(one, range(flat.shape[0]), n_jobs=n_jobs,
                                   progress=progress, desc="cepstral discreteness"), float)
    return vals.reshape(scan) if scan and len(scan) == 2 else vals


def _lagrange_reduce(v1, v2):
    """2-D Lagrange–Gauss lattice reduction: shortest basis of the lattice
    generated by ``v1, v2`` (both length-2 arrays, Å)."""
    v1 = np.array(v1, float); v2 = np.array(v2, float)
    if v1 @ v1 > v2 @ v2:
        v1, v2 = v2, v1
    for _ in range(64):
        m = round((v1 @ v2) / (v1 @ v1))
        v2 = v2 - m * v1
        if v2 @ v2 >= v1 @ v1:
            break
        v1, v2 = v2, v1
    return v1, v2


def _ewpc_peaks(cep, dr, r_min, r_max, nsig=4.0, min_sep_px=2, bg_win_px=None,
                min_prom=0.10, max_peaks=60, subpixel=True):
    """Detect discrete peaks in an EWPC pattern within the quefrency band.

    The EWPC decays steeply from the centre, so peaks are found on a
    *background-subtracted* map: ``resid = cep - median_filter(cep, bg_win)``.
    A peak must (i) be a local maximum over a ``±min_sep_px`` neighbourhood,
    (ii) exceed ``median + nsig·MAD`` of the residual in the band (robust
    significance, not a fraction of the global max), and (iii) exceed
    ``min_prom`` of the strongest residual peak. Positions are refined to
    sub-pixel by a parabolic fit in x and y — essential for accurate lattice
    vectors. Returns ``(vecs, strengths)`` with ``vecs`` the (dx, dy) offsets
    from the centre in Å, sorted by increasing length.
    """
    from scipy.ndimage import maximum_filter, median_filter

    n0, n1 = cep.shape
    if bg_win_px is None:
        bg_win_px = max(5, int(round(0.5 / dr)))
    if bg_win_px % 2 == 0:
        bg_win_px += 1
    resid = cep - median_filter(cep, size=int(bg_win_px))
    r_ang = _radius_px(cep.shape) * dr
    band = (r_ang >= r_min) & (r_ang <= r_max)
    vb = resid[band]
    med = float(np.median(vb))
    mad = 1.4826 * float(np.median(np.abs(vb - med))) + 1e-12
    thr = med + float(nsig) * mad
    sep = max(1, int(min_sep_px))
    loc = band & (resid == maximum_filter(resid, size=2 * sep + 1)) & (resid > thr)
    ys, xs = np.nonzero(loc)
    if ys.size == 0:
        return np.zeros((0, 2)), np.zeros(0)
    st = resid[ys, xs]
    keep = st >= float(min_prom) * st.max()
    ys, xs, st = ys[keep], xs[keep], st[keep]
    order = np.argsort(-st)[:int(max_peaks)]
    ys, xs, st = ys[order], xs[order], st[order]
    cy, cx = n0 // 2, n1 // 2
    fy = ys.astype(float); fx = xs.astype(float)
    if subpixel:
        for i in range(len(ys)):
            y, x = int(ys[i]), int(xs[i])
            if 0 < y < n0 - 1:
                a, b, c = resid[y - 1, x], resid[y, x], resid[y + 1, x]
                d = a - 2 * b + c
                if d < 0:
                    fy[i] = y + 0.5 * (a - c) / d
            if 0 < x < n1 - 1:
                a, b, c = resid[y, x - 1], resid[y, x], resid[y, x + 1]
                d = a - 2 * b + c
                if d < 0:
                    fx[i] = x + 0.5 * (a - c) / d
    vecs = np.column_stack([(fx - cx) * dr, (fy - cy) * dr])
    norms = np.hypot(vecs[:, 0], vecs[:, 1])
    o = np.argsort(norms)
    return vecs[o], st[o]


def _best_lattice_basis(vecs, weights=None, min_angle=15.0, lattice_tol=0.18,
                        n_basis=8, reduce=True):
    """Pick the primitive 2-D basis ``(v1, v2)`` of a point set that a lattice best
    explains. Searches all pairs of the ``n_basis`` shortest points (each Lagrange-
    reduced) and returns ``(frac, v1, v2)`` maximising the **strength-weighted**
    fraction of points whose fractional coordinates ``[h,k] = p·V⁻¹`` fall within
    ``lattice_tol`` of integers. Returns ``None`` if no non-collinear pair. Shared by
    the cepstral-peak lattice and the diffraction-spot lattice.

    With ``reduce=False`` the basis is kept as the two actual (shortest) input
    points rather than their Lagrange-reduced combination — so the returned
    vectors land exactly on detected spots (the reduced basis can be a shorter
    lattice vector that is not itself a detected point)."""
    vecs = np.asarray(vecs, float)
    if len(vecs) < 2:
        return None
    w = np.ones(len(vecs)) if weights is None else np.asarray(weights, float)
    w = w / (w.sum() + 1e-12)
    norms = np.hypot(vecs[:, 0], vecs[:, 1])
    cand = vecs[np.argsort(norms)][:min(len(vecs), int(n_basis))]
    best = None
    for a in range(len(cand)):
        for b in range(a + 1, len(cand)):
            v1c, v2c = cand[a], cand[b]
            cth = float(v1c @ v2c) / (np.linalg.norm(v1c) * np.linalg.norm(v2c) + 1e-12)
            if np.degrees(np.arccos(np.clip(abs(cth), -1.0, 1.0))) < min_angle:
                continue
            r1, r2 = _lagrange_reduce(v1c, v2c) if reduce else (v1c, v2c)
            V = np.array([r1, r2])
            if abs(np.linalg.det(V)) < 1e-9:
                continue
            hk = vecs @ np.linalg.inv(V)
            fit = np.abs(hk - np.round(hk)).max(axis=1) <= lattice_tol
            frac = float(w[fit].sum())
            if best is None or frac > best[0]:
                best = (frac, r1, r2)
    return best


def spot_lattice(spots, center, q_per_px, weights=None, min_angle=15.0,
                 lattice_tol=0.06, n_basis=8, hk_max=2, reduce=True):
    """Reciprocal lattice from **diffraction spots** — the direct, beam-robust route.

    The detected Bragg spots ARE the reciprocal lattice, so this skips the cepstral
    (and its central-beam blob) entirely. Converts ``spots`` ``[(x,y,q),...]`` to
    g-vectors, finds the primitive reciprocal basis ``g1, g2`` (1/Å) via
    :func:`_best_lattice_basis` (weighted by ``weights`` if given, e.g. spot
    intensity), and returns ``(info)`` with ``g1, g2, lattice_frac, gvectors`` and a
    generated reflection grid ``grid_gs = h·g1 + k·g2`` for indexing/plotting.
    ``lattice_frac`` (weighted fraction of spots on the 2-D lattice) is 1 for a good
    zone-axis pattern; a collinear 2-spot pair gives no 2-D basis (``None``) — that
    is the off-zone case, handled by |q| matching instead. Returns ``None`` if fewer
    than 2 non-collinear spots."""
    from .indexing import spots_to_gvectors
    if len(spots) < 2:
        return None
    gv = np.asarray(spots_to_gvectors(spots, center, q_per_px), float)
    best = _best_lattice_basis(gv, weights=weights, min_angle=min_angle,
                               lattice_tol=lattice_tol, n_basis=n_basis, reduce=reduce)
    if best is None:
        return None
    frac, g1, g2 = best
    grid = np.array([h * g1 + k * g2
                     for h in range(-hk_max, hk_max + 1)
                     for k in range(-hk_max, hk_max + 1)
                     if not (h == 0 and k == 0)])
    return {"g1": g1, "g2": g2, "lattice_frac": frac, "gvectors": gv, "grid_gs": grid}


def cepstral_periodicity(pattern, g1, g2=None, q_per_px=None, pad=None,
                         highpass="auto", n_orders=3, half_width=0.6, step=0.25,
                         smooth=1.0, tang_min=0.4, require_tangential=True,
                         snap_win=0.0):
    """Translational-periodicity test — the defining crystallinity property.

    A crystal has long-range translational order: its cepstral (EWPC) shows an
    *isolated* peak not only at the lattice vector ``a`` but at ``2a, 3a, …`` — the
    vector *repeats*. This cannot be faked by matching (which only checks the first
    application, satisfied by amorphous too); only repetition, checked here, needs a
    crystal.

    Rather than pre-detecting 2-D peaks and matching (fragile), each predicted point
    ``n·a_i`` is **verified from line profiles** through it. A genuine lattice peak is
    a maximum — **1st derivative ≈ 0 (stationary) AND 2nd derivative < 0 (concave)** —
    BOTH along the vector (radial) AND perpendicular to it (tangential). The
    stationarity is tested as ``|d1/d2| ≤ peak_tol`` (the parabola vertex lands within
    tolerance of the predicted point). An amorphous ring is a maximum radially too —
    it has cepstral shells at ``a, 2a, …`` from the halo harmonics — but is **flat
    tangentially** (``d2 ≈ 0``, the ring is extended along the angle), so the
    tangential curvature separates a lattice *point* from a *ring*. The tangential
    curvature (``−d2``) is normalised by the radial curvature of the first-order peak.
    A point *applies* when it is a maximum radially AND tangentially with normalised
    tangential curvature ≥ ``tang_min``; a vector *repeats* when it applies at 1st
    order AND at 2nd or 3rd. ``score`` is the
    min repeat strength over the basis vectors (0 if any fails): crystal ⇒ ≈ 1+,
    amorphous ⇒ ≈ 0. Returns ``{score, orders, a_vectors, dr}`` where ``orders[i]``
    lists ``(tang_norm, applies)`` for ``n = 1…n_orders``.
    """
    hp = highpass
    if hp == "auto":
        hp = max(4, int(np.asarray(pattern).shape[0]) // 32)
    cep = ewpc_pattern(pattern, pad=pad, highpass=hp)
    dr = quefrency_per_px(cep.shape[0], q_per_px)

    g1 = np.asarray(g1, float)
    if g2 is not None:
        G = np.array([g1, np.asarray(g2, float)])
        avs = [np.linalg.inv(G).T[0], np.linalg.inv(G).T[1]]  # a_i·g_j = δ_ij
    else:
        avs = [g1 / (g1 @ g1)]                                # 1-D: |a| = 1/|g|

    score, orders = _periodicity_from_cep(cep, dr, avs, n_orders=n_orders,
                                          half_width=half_width, step=step,
                                          smooth=smooth, tang_min=tang_min,
                                          require_tangential=require_tangential,
                                          snap_win=snap_win)
    return {"score": score, "orders": orders, "a_vectors": avs, "dr": dr}


def _periodicity_from_cep(cep, dr, avs, n_orders=3, half_width=0.6, step=0.25,
                          smooth=1.0, tang_min=0.4, require_tangential=True,
                          snap_win=0.0):
    """Translational-periodicity score of a *precomputed* EWPC ``cep`` for the
    real-space lattice vectors ``avs`` (Å). Shared core of
    :func:`cepstral_periodicity` (vector supplied from reciprocal g) and
    :func:`cepstral_periodicity_map` (vector read off the cepstral peaks). Each
    predicted point ``n·a_i`` is verified from line profiles: a genuine lattice
    peak is stationary (1st deriv ≈ 0) AND concave (2nd deriv < 0) BOTH radially
    (along a) and tangentially (perpendicular). An amorphous ring is concave
    radially but flat tangentially, so the tangential curvature — normalised by
    the 1st-order radial curvature — separates a lattice *point* from a *ring*.
    Returns ``(score, orders)``; ``score`` is the min repeat strength over the
    basis vectors (0 if any fails to repeat)."""
    from scipy.ndimage import map_coordinates, gaussian_filter1d
    n = cep.shape[0]
    cc = n // 2

    def deriv(origin_px, dir_hat):
        # 1st and 2nd derivative of the line profile at origin along dir_hat.
        # A peak here needs d1 ≈ 0 (stationary) AND d2 < 0 (concave-down).
        ts = np.arange(-half_width, half_width + 1e-9, step * dr)
        xs = origin_px[0] + ts / dr * dir_hat[0]
        ys = origin_px[1] + ts / dr * dir_hat[1]
        if xs.min() < 1 or xs.max() > n - 2 or ys.min() < 1 or ys.max() > n - 2:
            return None
        p = gaussian_filter1d(map_coordinates(cep, [ys, xs], order=1), smooth)
        d1 = np.gradient(p, ts)
        d2 = np.gradient(d1, ts)
        i0 = len(ts) // 2
        return float(d1[i0]), float(d2[i0])

    def is_peak(d, tol):
        # d = (d1, d2). Peak: stationary (|vertex offset| = |d1/d2| <= tol) AND concave (d2 < 0).
        if d is None:
            return False, 0.0
        d1, d2 = d
        if d2 >= 0:
            return False, 0.0                                # not concave -> not a maximum
        stationary = abs(d1 / d2) <= tol                     # peak vertex within tol of the point
        return bool(stationary), -d2                          # curvature magnitude (concavity)

    def snap(origin_px, dir_hat):
        # "if NEAR a peak, take it": snap the predicted point to the nearest
        # cepstral local maximum within +/- snap_win (A) along dir_hat. Tolerates
        # vector/calibration error so a peak slightly off the prediction is not lost.
        if snap_win <= 0:
            return origin_px
        ts = np.arange(-snap_win, snap_win + 1e-9, step * dr)
        xs = origin_px[0] + ts / dr * dir_hat[0]
        ys = origin_px[1] + ts / dr * dir_hat[1]
        if xs.min() < 1 or xs.max() > n - 2 or ys.min() < 1 or ys.max() > n - 2:
            return origin_px
        p = gaussian_filter1d(map_coordinates(cep, [ys, xs], order=1), smooth)
        lm = [k for k in range(1, len(p) - 1) if p[k] >= p[k - 1] and p[k] >= p[k + 1]]
        if not lm:
            return origin_px
        k = lm[int(np.argmin([abs(ts[j]) for j in lm]))]     # nearest local max to prediction
        return (origin_px[0] + ts[k] / dr * dir_hat[0], origin_px[1] + ts[k] / dr * dir_hat[1])

    peak_tol = 0.5 * half_width                               # vertex must be within ~half the window
    orders = {}
    rep = []
    for i, a in enumerate(avs):
        L = float(np.hypot(*a))
        ah = a / L
        tg = np.array([-ah[1], ah[0]])                       # perpendicular (tangential)
        _, ref = is_peak(deriv(snap((cc + a[0] / dr, cc + a[1] / dr), ah), ah), peak_tol)  # 1st-order radial curvature
        ref = ref if ref > 0 else None
        os = []
        for ns in range(1, int(n_orders) + 1):
            o = snap((cc + ns * a[0] / dr, cc + ns * a[1] / dr), ah)   # snap to nearby cepstral peak
            pr, cr = is_peak(deriv(o, ah), peak_tol)         # radial:  1st deriv 0, 2nd deriv < 0
            pt, ct = is_peak(deriv(o, tg), peak_tol)         # tangential: 1st deriv 0, 2nd deriv < 0
            if ref is None:
                os.append((0.0, False)); continue
            tnorm = ct / ref                                 # isolated point -> ~1; ring -> ~0 (flat tangentially)
            if require_tangential:
                ap = bool(pr and pt and tnorm >= tang_min)   # isolated 2-D lattice point
            else:
                ap = bool(pr)                                # radial only (1-D / collinear: peaks along the vector only)
            os.append((round(tnorm, 2), ap))
        orders[i] = os
        if not os[0][1]:                                     # must apply once (isolated 2-D peak)
            rep.append(0.0)
            continue
        higher = [o[0] for o in os[1:] if o[1]]              # repeats at 2nd OR 3rd
        rep.append(max(higher) if higher else 0.0)
    score = float(min(rep)) if rep else 0.0
    return score, orders


def cepstral_periodicity_map(cube, center=None, q_per_px=None, mask=None,
                             r_min=1.0, r_max=8.0, highpass="auto", pad=None,
                             offset=1.0, peak_nsig=4.0, lattice_tol=0.18,
                             min_angle=15.0, n_orders=3,
                             half_width=0.6, tang_min=0.4, two_d=True,
                             scan_bin=1, n_jobs=1, progress=False):
    """Exhaustive translational-periodicity crystallinity map over the scan.

    The *mathematical* crystallinity test of §7c applied to **every** (masked)
    scan region, independent of any spot picking. For each NBD: the EWPC is
    formed (central beam high-passed), a PRIMITIVE lattice basis is read from
    its cepstral peaks (via :func:`_best_lattice_basis`), and
    :func:`_periodicity_from_cep` scores whether those vectors *repeat* — an
    isolated (radial **and** tangential) peak at ``2a, 3a`` marks long-range
    translational order (crystal), while an amorphous ring repeats radially but
    is flat tangentially and scores ≈ 0. Positions outside ``mask`` are 0.

    ``scan_bin`` sets the tested unit: with ``scan_bin=1`` every probe position
    is tested individually, but a *single* probe's NBD is low-dose and the
    cepstral peaks that build the vector are then SNR-limited (on real data
    almost every single pixel scores 0). Set ``scan_bin=k`` to average each
    ``k×k`` block of probe positions into one higher-dose NBD before the test —
    still exhaustive (every pixel belongs to a block, and the block score is
    written back to all its pixels) but with ``k²×`` the counts, which is what
    makes translational periodicity measurable on real amorphous-matrix data.

    Returns a scan-shaped map (high = crystalline). Pair with
    :func:`~fourdstem.analysis.indexing.label_grains` → ``grain_patterns`` →
    ``index_pattern`` to name the phase of each grain.
    """
    from .virtual_image import _resolve_center
    from ..utils.parallel import parallel_map
    center = _resolve_center(cube, center)
    if q_per_px is None:
        q_per_px = cube.calibration.q_per_px
    flat = cube._flat_patterns()
    H, W = flat.shape[1], flat.shape[2]
    scan = cube.scan_shape
    k = max(1, int(scan_bin))

    # Build the working patterns (single pixels, or k×k block averages) and the
    # index map from every scan pixel to its working unit.
    if k > 1 and scan and len(scan) == 2:
        Ry, Rx = scan
        by, bx = Ry // k, Rx // k
        cut_y, cut_x = by * k, bx * k
        blk = flat[:Ry * Rx].reshape(Ry, Rx, H, W)[:cut_y, :cut_x]
        work = blk.reshape(by, k, bx, k, H, W).mean(axis=(1, 3)).reshape(by * bx, H, W)
        if mask is not None:
            mm = np.asarray(mask, bool)[:cut_y, :cut_x].reshape(by, k, bx, k).mean(axis=(1, 3)) >= 0.5
            mwork = mm.ravel()
        else:
            mwork = None
    else:
        work = flat
        mwork = None if mask is None else np.asarray(mask, bool).ravel()
        by = bx = None

    hp = highpass
    if hp == "auto":
        hp = max(4, H // 32)
    dr = quefrency_per_px(ewpc_pattern(work[0], offset=offset, pad=pad,
                                       highpass=hp).shape[0], q_per_px)
    bg_win = max(5, int(round(0.5 / dr)))

    def _score(i):
        if mwork is not None and not mwork[i]:
            return 0.0
        try:
            cep = ewpc_pattern(work[i], offset=offset, pad=pad, highpass=hp)
            vecs, st = _ewpc_peaks(cep, dr, r_min, r_max, nsig=peak_nsig,
                                   bg_win_px=bg_win)
            if len(vecs) < 2:
                return 0.0
            # PRIMITIVE basis from the cepstral peaks (shortest pair explaining
            # the most peaks) — not the single strongest peak, which is often a
            # diagonal 2a-type vector that fails to repeat.
            best = _best_lattice_basis(vecs, weights=st, min_angle=min_angle,
                                       lattice_tol=lattice_tol, n_basis=8)
            if best is None:
                return 0.0
            _frac, v1, v2 = best
            avs = [np.asarray(v1, float), np.asarray(v2, float)] if two_d \
                else [np.asarray(v1, float)]
            score, _ = _periodicity_from_cep(cep, dr, avs, n_orders=n_orders,
                                             half_width=half_width, tang_min=tang_min)
            return float(score)
        except Exception:
            return 0.0

    vals = np.asarray(parallel_map(_score, range(work.shape[0]), n_jobs=n_jobs,
                                   progress=progress, desc="periodicity"), float)
    if k > 1 and scan and len(scan) == 2:
        out = np.zeros(scan, float)
        up = np.repeat(np.repeat(vals.reshape(by, bx), k, 0), k, 1)  # block -> pixels
        out[:by * k, :bx * k] = up
        return out
    return vals.reshape(scan) if scan else vals


def cepstral_angular_map(cube, center=None, q_per_px=None, mask=None,
                         r_min=1.0, r_max=6.0, highpass="auto", offset=1.0,
                         ring_hw=1.5, n_jobs=1, progress=False):
    """Exhaustive per-pixel cepstral crystallinity LOCATOR (angular concentration).

    The crystallinity test of §7c made *robust per single low-dose probe* so it
    can locate crystalline regions over the whole scan. Same physics as the
    translational-periodicity test — a crystal's EWPC is discrete **points**, an
    amorphous phase's EWPC is a **ring** — but read *around* the dominant
    quefrency shell instead of stepping out to ``2a, 3a``. For each NBD: form the
    EWPC (central beam high-passed, **no zero-padding needed**), find the
    strongest shell radius ``r*`` in the band ``[r_min, r_max]`` Å from the radial
    cepstrum, and score the azimuthal concentration on the annulus ``r* ± ring_hw``
    as ``(max − median) / MAD``. A few discrete lattice points give a high value;
    a full ring is azimuthally flat and scores low. Because it is a robust ratio
    (not a fragile 2-D peak test) and needs no padding, it works where the
    per-pixel :func:`cepstral_periodicity_map` collapses to ~0 on real low-dose
    data. Use it to *locate* grains, then confirm and index them with
    :func:`cepstral_periodicity` on the grain-mean NBD (where the dose is enough
    for the rigorous 2a/3a repeat). Positions outside ``mask`` are 0.

    Returns a scan-shaped map (high = crystalline / spotty cepstrum).
    """
    from .virtual_image import _resolve_center
    from ..utils.parallel import parallel_map
    center = _resolve_center(cube, center)
    if q_per_px is None:
        q_per_px = cube.calibration.q_per_px
    flat = cube._flat_patterns()
    N = flat.shape[0]
    H, W = flat.shape[1], flat.shape[2]
    hp = highpass
    if hp == "auto":
        hp = max(4, H // 32)
    n0 = ewpc_pattern(flat[0], offset=offset, highpass=hp).shape[0]
    cc = n0 // 2
    dr = quefrency_per_px(n0, q_per_px)
    yy, xx = np.mgrid[0:n0, 0:n0]
    rad = np.hypot(xx - cc, yy - cc)
    m = None if mask is None else np.asarray(mask, bool).ravel()

    def _score(i):
        if m is not None and not m[i]:
            return 0.0
        try:
            cep = ewpc_pattern(flat[i], offset=offset, highpass=hp)
            r, prof = cepstral_radial_profile(cep, q_per_px, r_min=r_min)
            band = (r >= r_min) & (r <= r_max)
            if band.sum() < 2 or prof[band].max() <= 0:
                return 0.0
            rstar = r[band][int(np.argmax(prof[band]))]
            rp = rstar / dr
            ann = (rad >= rp - ring_hw) & (rad <= rp + ring_hw)
            v = cep[ann]
            if v.size < 8:
                return 0.0
            med = float(np.median(v))
            mad = 1.4826 * float(np.median(np.abs(v - med))) + 1e-9
            return float((float(v.max()) - med) / mad)
        except Exception:
            return 0.0

    vals = np.asarray(parallel_map(_score, range(N), n_jobs=n_jobs,
                                   progress=progress, desc="cepstral angular"), float)
    scan = cube.scan_shape
    return vals.reshape(scan) if scan else vals


def cepstral_spot_periodicity_map(cube, center=None, q_per_px=None, mask=None,
                                  q_beam=0.20, q_max=1.15, tophat=11, n_mad=4.5,
                                  keep=6, log_spot=True, log_off=1.0,
                                  lattice_tol=0.06, hk_max=2, min_angle=15.0,
                                  pad=None, highpass="auto", n_orders=3,
                                  half_width=0.6, tang_min=0.4, n_jobs=1,
                                  progress=False):
    """Exhaustive per-pixel run of the full §7c spot→vector→periodicity chain.

    For **every** (masked) probe position, exactly the region-level recipe:
      1. detect Bragg spots on ``log(NBD)`` (log lifts weak high-order spots and
         flattens the amorphous ring), keep the strongest ``keep``;
      2. build the reciprocal lattice ``g1, g2`` from those spots
         (:func:`spot_lattice`), i.e. the reciprocal→real vectors;
      3. run :func:`cepstral_periodicity` — convert to the real lattice, and
         verify the vector *repeats* at ``2a, 3a`` by the line-profile test
         (1st derivative ≈ 0 AND 2nd derivative < 0, radial and tangential).
    Collinear spots fall back to the 1-D single-vector periodicity. Returns the
    per-pixel periodicity-score map.

    This is the user's exact method applied exhaustively. **Expect it to be
    sparse on real data**: a single probe is low-dose and rarely shows the ≥2
    spots step 1 needs, so most pixels score 0 — the physical SNR limit, not a
    bug. For a dense verdict, average within grains and apply the same chain to
    the grain mean (see :func:`cepstral_angular_map` for a robust locator).
    """
    from .virtual_image import _resolve_center
    from ..utils.parallel import parallel_map
    from .phases import detect_spots
    from .indexing import spots_to_gvectors
    center = _resolve_center(cube, center)
    if q_per_px is None:
        q_per_px = cube.calibration.q_per_px
    flat = cube._flat_patterns()
    N = flat.shape[0]
    H, W = flat.shape[1], flat.shape[2]
    cen = (float(center[0]), float(center[1]))
    m = None if mask is None else np.asarray(mask, bool).ravel()

    def _val(nbd, s):
        return float(nbd[int(np.clip(s[1], 0, H - 1)), int(np.clip(s[0], 0, W - 1))])

    def _score(i):
        if m is not None and not m[i]:
            return 0.0
        try:
            nbd = np.asarray(flat[i], float)
            det = np.log(np.clip(nbd, 0, None) + log_off) if log_spot else nbd
            sp = detect_spots(det, cen, q_per_px, q_beam=q_beam, q_max=q_max,
                              n_mad=n_mad, min_dist=3, tophat=tophat)
            if len(sp) < 2:
                return 0.0
            sp = [s for _, s in sorted(((_val(det, s), s) for s in sp),
                                       key=lambda t: -t[0])[:keep]]
            wt = [_val(nbd, s) for s in sp]
            lat = spot_lattice(sp, cen, q_per_px, weights=wt,
                               lattice_tol=lattice_tol, hk_max=hk_max,
                               min_angle=min_angle)
            if lat is not None:
                per = cepstral_periodicity(nbd, lat["g1"], lat["g2"], q_per_px,
                                           pad=pad, highpass=highpass,
                                           n_orders=n_orders, half_width=half_width,
                                           tang_min=tang_min)
            else:                                    # collinear -> 1-D periodicity
                gv = spots_to_gvectors(sp, cen, q_per_px)
                gs = gv[int(np.argmax(wt))]
                per = cepstral_periodicity(nbd, gs, None, q_per_px, pad=pad,
                                           highpass=highpass, n_orders=n_orders,
                                           half_width=half_width, tang_min=tang_min)
            return float(per["score"])
        except Exception:
            return 0.0

    vals = np.asarray(parallel_map(_score, range(N), n_jobs=n_jobs,
                                   progress=progress, desc="spot-periodicity"), float)
    scan = cube.scan_shape
    return vals.reshape(scan) if scan else vals


def cepstral_lattice_gvectors(pattern, q_per_px, r_min=1.0, r_max=6.0, pad=None,
                              offset=1.0, window=True, min_prom=0.10, min_angle=15.0,
                              hk_max=2, max_peaks=60, min_lattice_frac=0.5,
                              lattice_tol=0.18, peak_nsig=8.0, min_sep_px=2,
                              bg_frac=0.5, n_basis=8, highpass="auto"):
    """Extract the 2-D EWPC peak lattice and return equivalent reciprocal g-vectors.

    The exit-wave power cepstrum of a crystalline zone-axis pattern peaks at the
    projected *real-space* lattice vectors. Peak detection (the make-or-break
    step) is done by :func:`_ewpc_peaks`: background-subtracted, robust
    ``median + peak_nsig·MAD`` significance, ``min_sep_px`` separation and
    sub-pixel parabolic refinement (``bg_frac`` sets the background window in Å).
    The primitive basis ``v1, v2`` (Å) is then chosen by searching all pairs of
    the ``n_basis`` shortest peaks (each Lagrange-reduced) and keeping the one
    whose lattice explains the most peaks — so a single spurious short peak cannot
    wreck the basis. ``lattice_frac`` is that best fraction, **weighted by peak
    strength**: the summed strength of peaks whose fractional coordinates
    ``[h,k] = p·V⁻¹`` land within ``lattice_tol`` of integers, over the total. The
    weighting matters because the residual central-beam blob leaves many *weak*
    ripple peaks — unweighted they would swamp the few strong true-lattice peaks. A
    crystal scores ≈1; an amorphous cepstral *ring* scores low and, below
    ``min_lattice_frac``, returns an empty ``gs``. From ``v1, v2`` the 2-D
    reciprocal basis ``g1, g2`` (1/Å, ``g_i · v_j = δ_ij`` so ``|g|=1/d``) gives the
    reflection set ``g_hk = h·g1 + k·g2`` (``|h|,|k| ≤ hk_max``) for
    :func:`index_pattern`. Returns ``(gs, info)`` with ``info`` holding
    ``v1, v2, g1, g2, lattice_frac, peaks, peak_strength, dr, n_peaks`` (and
    ``reason`` when empty). ``peaks`` are the detected (dx, dy) offsets in Å — plot
    them on the EWPC to check detection quality.
    """
    hp = highpass
    if hp == "auto":
        hp = max(4, int(np.asarray(pattern).shape[0]) // 32)   # ~beam+halo width
    cep = ewpc_pattern(pattern, offset=offset, window=window, pad=pad, highpass=hp)
    n0, n1 = cep.shape
    dr = quefrency_per_px(n0, q_per_px)
    bg_win = max(5, int(round(float(bg_frac) / dr)))
    vecs, st = _ewpc_peaks(cep, dr, r_min, r_max, nsig=peak_nsig,
                           min_sep_px=min_sep_px, bg_win_px=bg_win,
                           min_prom=min_prom, max_peaks=max_peaks)
    empty = np.zeros((0, 2))
    base_info = {"dr": dr, "n_peaks": int(len(vecs)),
                 "peaks": vecs, "peak_strength": st}
    if len(vecs) < 2:
        return empty, {"reason": "fewer than 2 peaks", **base_info}

    # Strength-weighted basis search (shared helper): strong true-lattice peaks
    # dominate; weak central-blob ripple peaks can't dilute the fraction.
    inv = np.linalg.inv
    best = _best_lattice_basis(vecs, weights=st, min_angle=min_angle,
                               lattice_tol=lattice_tol, n_basis=n_basis)
    if best is None:
        return empty, {"reason": "no non-collinear basis", **base_info}
    frac, v1, v2 = best
    if frac < float(min_lattice_frac):
        return empty, {"reason": f"peaks not lattice-consistent (frac={frac:.2f})",
                       "lattice_frac": frac, **base_info}
    G = inv(np.array([v1, v2])).T
    g1, g2 = G[0], G[1]
    gs = [h * g1 + k * g2
          for h in range(-hk_max, hk_max + 1)
          for k in range(-hk_max, hk_max + 1)
          if not (h == 0 and k == 0)]
    info = {"v1": v1, "v2": v2, "g1": g1, "g2": g2, "lattice_frac": frac,
            "peaks": vecs, "peak_strength": st, **base_info}
    return np.array(gs), info


def ewpc_mean(cube, offset=1.0, window=True, reducer="mean", n_jobs=1,
              progress=False):
    """Mean (or median) EWPC pattern over all probe positions — a representative
    cepstrum for display and for reading off inter-atomic distances."""
    from ..utils.parallel import parallel_map

    flat = cube._flat_patterns() if hasattr(cube, "_flat_patterns") else \
        np.asarray(cube).reshape(-1, *np.asarray(cube).shape[-2:])
    ceps = parallel_map(lambda i: ewpc_pattern(flat[i], offset=offset, window=window),
                        range(flat.shape[0]), n_jobs=n_jobs, progress=progress,
                        desc="EWPC")
    stack = np.stack(ceps)
    return np.median(stack, 0) if reducer == "median" else stack.mean(0)


def cepstral_peakiness_image(cube, r_in, r_out, q_per_px, offset=1.0, window=True,
                             pctl=99.5, mask=None, n_jobs=1, progress=False):
    """Per-pixel EWPC **peak prominence** over the quefrency band ``[r_in, r_out]`` Å.

    A *direct* crystalline/amorphous discriminator matching the physics: a
    crystalline pattern's EWPC has a few **sharp discrete peaks** (the lattice in
    real space), while an amorphous pattern's EWPC is a **diffuse blob**. For each
    probe position this computes the strongest cepstral peak in the band relative
    to its local background,

        peakiness = (percentile(C, ``pctl``) - median(C)) / MAD(C),

    over the band pixels ``C`` — high when a sharp discrete peak stands above a
    flat background (crystalline), low for a smooth diffuse cepstrum (amorphous).
    Complements :func:`fluctuation_image` (normalized variance): the percentile /
    MAD form keys directly on *discrete peakiness* and is more robust to the broad
    amorphous speckle that muddies a plain variance. Returns a scan-shaped map.
    """
    from ..utils.parallel import parallel_map

    flat = cube._flat_patterns() if hasattr(cube, "_flat_patterns") else \
        np.asarray(cube).reshape(-1, *np.asarray(cube).shape[-2:])
    dp = flat.shape[1:]
    ann = _annulus_mask(dp, q_per_px, r_in, r_out)
    if mask is not None:
        ann = ann & np.asarray(mask, bool)
    if ann.sum() < 5:
        raise ValueError(
            f"annulus [{r_in},{r_out}] Å selects {int(ann.sum())} cepstral pixels; "
            f"widen the band or check q_per_px (dr="
            f"{quefrency_per_px(dp[0], q_per_px):.4g} Å/px).")

    def one(i):
        cep = ewpc_pattern(flat[i], offset=offset, subtract_mean=True, window=window)
        v = cep[ann]
        med = float(np.median(v))
        mad = 1.4826 * float(np.median(np.abs(v - med))) + 1e-12
        return float((np.percentile(v, pctl) - med) / mad)

    vals = np.asarray(parallel_map(one, range(flat.shape[0]), n_jobs=n_jobs,
                                   progress=progress, desc="EWPC peakiness"), float)
    scan = cube.scan_shape if hasattr(cube, "scan_shape") else None
    return vals.reshape(scan) if scan and len(scan) == 2 else vals


def ewpc_profiles(cube, q_per_px, n_bins=None, r_min=0.3, offset=1.0,
                  window=True, n_jobs=1, progress=False):
    """Per-position cepstral radial profiles -> ``(profiles, r_A)``.

    ``profiles`` is ``(n_pos, n_bins)`` — a compact per-probe structural feature
    (inter-atomic-distance signature) for clustering / NMF / PCA phase separation,
    computed without any background subtraction.
    """
    from ..utils.parallel import parallel_map

    flat = cube._flat_patterns() if hasattr(cube, "_flat_patterns") else \
        np.asarray(cube).reshape(-1, *np.asarray(cube).shape[-2:])
    r_ref, _ = cepstral_radial_profile(
        ewpc_pattern(flat[0], offset=offset, window=window), q_per_px,
        n_bins=n_bins, r_min=r_min)

    def one(i):
        cep = ewpc_pattern(flat[i], offset=offset, window=window)
        _, prof = cepstral_radial_profile(cep, q_per_px, n_bins=n_bins, r_min=r_min)
        return prof

    profs = parallel_map(one, range(flat.shape[0]), n_jobs=n_jobs,
                         progress=progress, desc="EWPC profiles")
    return np.stack(profs), r_ref
