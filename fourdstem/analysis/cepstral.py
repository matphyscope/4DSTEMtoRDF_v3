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
    "fluctuation_image", "fluctuation_multiband", "fluctuation_profile", "cepstral_peakiness_image", "cepstral_discreteness_image", "cepstral_lattice_gvectors", "ewpc_mean", "ewpc_profiles",
]


def ewpc_pattern(pattern, offset=1.0, subtract_mean=True, window=True, pad=None):
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
    """
    p = np.asarray(pattern, float)
    logI = np.log(np.maximum(p, 0.0) + offset)
    if subtract_mean:
        logI = logI - logI.mean()
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


def cepstral_lattice_gvectors(pattern, q_per_px, r_min=1.0, r_max=6.0, pad=None,
                              offset=1.0, window=True, min_prom=0.15, min_angle=15.0,
                              hk_max=2, max_peaks=40, min_lattice_frac=0.5, lattice_tol=0.22):
    """Extract the 2-D EWPC peak lattice and return equivalent reciprocal g-vectors.

    The exit-wave power cepstrum of a crystalline zone-axis pattern peaks at the
    projected *real-space* lattice vectors. This finds the two shortest
    non-collinear cepstral peaks (Lagrange-reduced to a primitive basis ``v1, v2``
    in Å), forms the 2-D reciprocal basis ``g1, g2`` (1/Å, ``g_i · v_j = δ_ij`` so
    ``|g|=1/d``), and generates ``g_hk = h·g1 + k·g2`` (``|h|,|k| ≤ hk_max``) — a
    clean reflection set to feed :func:`index_pattern` for geometric zone-axis
    indexing. Peaks below ``min_prom`` of the max are dropped; an amorphous pattern
    (cepstral *rings*, no discrete 2-D lattice) yields no non-collinear basis and
    returns an empty ``gs``. Returns ``(gs, info)`` with ``info`` holding
    ``v1, v2, g1, g2, dr, n_peaks`` (and ``reason`` when empty).
    """
    from scipy.ndimage import maximum_filter

    cep = ewpc_pattern(pattern, offset=offset, window=window, pad=pad)
    n0, n1 = cep.shape
    dr = quefrency_per_px(n0, q_per_px)
    cy, cx = n0 // 2, n1 // 2
    r_ang = _radius_px(cep.shape) * dr
    band = (r_ang >= r_min) & (r_ang <= r_max)
    loc = (cep == maximum_filter(cep, size=3)) & band
    empty = np.zeros((0, 2))
    if not loc.any():
        return empty, {"reason": "no cepstral peaks in band", "dr": dr, "n_peaks": 0}
    ys, xs = np.nonzero(loc)
    vals = cep[ys, xs]
    keep = vals >= vals.max() * float(min_prom)
    ys, xs, vals = ys[keep], xs[keep], vals[keep]
    order = np.argsort(-vals)[:max_peaks]
    ys, xs = ys[order], xs[order]
    vecs = np.column_stack([(xs - cx) * dr, (ys - cy) * dr])       # (dx, dy) Å
    norms = np.hypot(vecs[:, 0], vecs[:, 1])
    o = np.argsort(norms)
    vecs = vecs[o]
    if len(vecs) < 2:
        return empty, {"reason": "fewer than 2 peaks", "dr": dr, "n_peaks": len(vecs)}
    v1 = vecs[0]
    v2 = None
    for v in vecs[1:]:
        c = float(v1 @ v) / (np.linalg.norm(v1) * np.linalg.norm(v) + 1e-12)
        ang = np.degrees(np.arccos(np.clip(abs(c), -1.0, 1.0)))    # 0..90 (fold)
        if ang >= min_angle:
            v2 = v
            break
    if v2 is None:
        return empty, {"reason": "no non-collinear peak", "dr": dr, "n_peaks": len(vecs)}
    v1, v2 = _lagrange_reduce(v1, v2)
    V = np.array([v1, v2])
    if abs(np.linalg.det(V)) < 1e-9:
        return empty, {"reason": "degenerate basis", "dr": dr, "n_peaks": len(vecs)}
    # lattice-consistency gate: a real crystal lattice has most detected peaks at
    # near-integer (h,k) of the basis; amorphous cepstral rings do not.
    hk = vecs @ np.linalg.inv(V)                                   # fractional coords
    resid = np.abs(hk - np.round(hk)).max(axis=1)
    frac = float((resid <= lattice_tol).mean())
    if frac < float(min_lattice_frac):
        return empty, {"reason": f"peaks not lattice-consistent (frac={frac:.2f})",
                       "dr": dr, "n_peaks": len(vecs), "lattice_frac": frac}
    G = np.linalg.inv(V).T
    g1, g2 = G[0], G[1]
    gs = [h * g1 + k * g2
          for h in range(-hk_max, hk_max + 1)
          for k in range(-hk_max, hk_max + 1)
          if not (h == 0 and k == 0)]
    info = {"v1": v1, "v2": v2, "g1": g1, "g2": g2, "dr": dr,
            "n_peaks": len(vecs), "lattice_frac": frac}
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
