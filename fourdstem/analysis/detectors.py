"""
fourdstem.analysis.detectors
============================
Region finding by **vacuum-referenced annular detectors** — the sensitive
"where is there real signal" stage that precedes strict identification.

Place an annular (ring) detector at an expected radius, measure the peak there
*above its local background*, and compare it to the same measurement over the
**vacuum** (no-sample) positions: vacuum is the noise floor, so a pixel whose
signal rises a set number of sigma above it carries **real** scattering, not
noise. This is the honest "real vs fake" cutoff.

Used for:
  * **halo** (amorphous) detectors at the broad diffuse maxima (FSDP + secondary)
    -> where amorphous material actually is (material vs vacuum), no phase name.
  * **ring** (polycrystalline) detectors at each candidate's sharp Bragg-ring
    radii -> where that phase's rings are real; a region lighting up several of a
    phase's rings is a predicted region for it.

Everything is computed from one per-pixel radial stack (:func:`radial_stack`),
so many detectors are cheap.
"""
from __future__ import annotations
import numpy as np

from .classify import radial_stack


def strict_vacuum_mask(cube, center=None, q_per_px=None, pctl=15.0):
    """Conservative vacuum = the ``pctl`` % of positions with the least total
    scattering (definitely empty). Independent of any halo/material threshold, so
    it is a clean reference for dark levels and detector cutoffs."""
    from .virtual_image import _resolve_center
    center = _resolve_center(cube, center)
    flat = cube._flat_patterns()
    N = flat.shape[0]
    tot = np.asarray(flat.reshape(N, -1).sum(1), float)
    scan = cube.scan_shape
    m = tot <= np.percentile(tot, pctl)
    return m.reshape(scan) if scan else m


def peak_above_flank(q, prof, q0, dq, flank=2.0):
    """Annular signal at ``q0``, per position. ``prof`` is ``(Npix, nbin)``.

    Mean intensity in the detector band ``[q0-dq, q0+dq]``, optionally minus the
    mean over the two flanking bands (``dq..flank*dq`` on each side):

    * ``flank`` a number > 1 (default): **peak above local background** — a smooth
      (thickness) background subtracts to ~0, isolating a **sharp** feature (a
      polycrystalline ring). This is the RING detector.
    * ``flank`` ``None`` (or <= 1): **plain annular mean** — no flank subtraction,
      so a **broad** feature (an amorphous halo) is not cancelled by its own tails.
      This is the HALO detector; the vacuum reference (not the flanks) supplies the
      background there.
    """
    band = (q >= q0 - dq) & (q <= q0 + dq)
    if band.sum() == 0:
        return np.zeros(prof.shape[0])
    center = prof[:, band].mean(1)
    if flank is None or flank <= 1:
        return center
    lo = (q >= q0 - flank * dq) & (q < q0 - dq)
    hi = (q > q0 + dq) & (q <= q0 + flank * dq)
    flanks = []
    if lo.any():
        flanks.append(prof[:, lo].mean(1))
    if hi.any():
        flanks.append(prof[:, hi].mean(1))
    base = np.mean(flanks, axis=0) if flanks else 0.0
    return center - base


def significance(vals, vacuum_mask):
    """``(vals - median_vacuum) / MAD_std_vacuum`` per position — sigma above the
    vacuum noise floor. ``vals`` and ``vacuum_mask`` share the scan shape."""
    vals = np.asarray(vals, float)
    v = vals[np.asarray(vacuum_mask, bool)]
    if v.size == 0:
        return np.zeros_like(vals)
    med = float(np.median(v))
    mad = 1.4826 * float(np.median(np.abs(v - med)))
    return (vals - med) / (mad if mad > 0 else (v.std() + 1e-9))


def detector_map(cube, center, q_per_px, q0, dq=0.03, flank=2.0, vacuum_mask=None,
                 q_max=1.2, nbin=200, n_jobs=1, _stack=None):
    """Vacuum-referenced significance map for a ring detector at ``q0`` (1/A).

    Returns ``(sig, raw)``: ``sig`` = sigma above vacuum (:func:`significance` of
    the :func:`peak_above_flank` height), ``raw`` = the peak height itself, both
    over the scan grid. Pass ``_stack=(q, prof)`` to reuse a radial stack across
    detectors (cheap). ``vacuum_mask`` defaults to a strict low-intensity vacuum.
    """
    scan = cube.scan_shape
    if _stack is None:
        q, prof = radial_stack(cube, center, q_per_px, q_max=q_max, nbin=nbin, n_jobs=n_jobs)
    else:
        q, prof = _stack
    raw = peak_above_flank(q, prof, q0, dq, flank=flank)
    if vacuum_mask is None:
        vacuum_mask = strict_vacuum_mask(cube, center=center, q_per_px=q_per_px)
    raw2d = raw.reshape(scan)
    return significance(raw2d, vacuum_mask), raw2d


def halo_bump_maps(cube, center, q_per_px, halo_qs, dq=0.05, beam_cut=0.15,
                   q_max=1.1, deg=2, vacuum_mask=None, nbin=200, n_jobs=1, _stack=None):
    """Bump maps at several halo radii using ONE global smooth background per pixel.

    The right way to isolate a broad halo near the beam: fit a smooth curve (a
    degree-``deg`` polynomial in log-log, i.e. a flexible power law — the beam tail
    + small-angle background) to the pixel's radial profile over the **background
    q-points** (everything above ``beam_cut`` and outside a ``±dq`` window of each
    halo radius), subtract it, and read the leftover **bump** in each halo band.
    Unlike flank lines this uses points on BOTH the low side (just above the beam)
    and the high side, so it works for a low-q FSDP where a wide flank would fall
    into the beam; unlike material-minus-vacuum it removes the material's OWN
    background, not just the vacuum's.

    ``halo_qs`` is the list of halo radii (1/A). Returns ``{q: dict(sig, raw,
    contrast, csig, bg)}`` — ``raw`` the bump height and ``sig`` its
    vacuum-referenced significance (both scale with thickness); ``contrast`` =
    bump / local-background (the **thickness-independent** structural indicator:
    a thick but featureless pixel has a big raw bump only from fit wiggle, but a
    small contrast), and ``csig`` its vacuum-referenced significance — **use
    ``csig`` for the structural-halo map** (a real halo clears vacuum by several
    sigma while thick-flat material does not). ``bg`` is the fitted background
    level in the band.
    """
    scan = cube.scan_shape
    if _stack is None:
        q, prof = radial_stack(cube, center, q_per_px, q_max=q_max + 0.05, nbin=nbin, n_jobs=n_jobs)
    else:
        q, prof = _stack
    halo_qs = [float(h) for h in halo_qs]
    bg = (q >= beam_cut) & (q <= q_max)
    for h in halo_qs:
        bg &= ~((q >= h - dq) & (q <= h + dq))
    if bg.sum() < deg + 1:
        raise ValueError("not enough background q-points to fit; widen beam_cut/q_max")
    lq = np.log(np.clip(q, 1e-6, None))
    X = np.vander(lq[bg], deg + 1)                     # (nbg, deg+1) design in log q
    Xp = np.linalg.pinv(X)                             # (deg+1, nbg)
    LP = np.log(np.clip(prof[:, bg], 1e-3, None))      # (Npix, nbg)
    coef = LP @ Xp.T                                   # (Npix, deg+1)
    Xall = np.vander(lq, deg + 1)                       # (nbin, deg+1)
    bgfit = np.exp(coef @ Xall.T)                       # (Npix, nbin) smooth background
    resid = prof - bgfit
    if vacuum_mask is None:
        vacuum_mask = strict_vacuum_mask(cube, center=center, q_per_px=q_per_px)
    out = {}
    for h in halo_qs:
        band = (q >= h - dq) & (q <= h + dq)
        if band.any():
            raw = np.clip(resid[:, band], 0, None).mean(1)
            bglev = np.clip(bgfit[:, band].mean(1), 1e-6, None)
            contrast = raw / bglev
        else:
            raw = np.zeros(prof.shape[0]); bglev = np.ones(prof.shape[0]); contrast = raw.copy()
        raw = raw.reshape(scan); contrast = contrast.reshape(scan); bglev = bglev.reshape(scan)
        out[h] = dict(sig=significance(raw, vacuum_mask), raw=raw,
                      contrast=contrast, csig=significance(contrast, vacuum_mask),
                      bg=bglev)
    return out


def halo_background_1d(q, prof1d, halo_qs, dq=0.05, beam_cut=0.15, q_max=1.1, deg=2):
    """Smooth convex background of a single radial profile — the same log-log
    power-law fit :func:`halo_bump_maps` uses per pixel, but for one 1D profile
    (e.g. the material-mean) so the demo picture matches the map. Fit over
    ``[beam_cut, q_max]`` minus a ``±dq`` window around each halo radius; returns
    the background evaluated at every ``q`` (halo shoulders rise above it)."""
    q = np.asarray(q, float); prof1d = np.asarray(prof1d, float)
    m = (q >= beam_cut) & (q <= q_max)
    for h in halo_qs:
        m &= ~((q >= h - dq) & (q <= h + dq))
    if m.sum() < deg + 1:
        raise ValueError("not enough background q-points; widen beam_cut/q_max")
    lq = np.log(np.clip(q, 1e-6, None))
    coef = np.linalg.lstsq(np.vander(lq[m], deg + 1),
                           np.log(np.clip(prof1d[m], 1e-3, None)), rcond=None)[0]
    return np.exp(np.vander(lq, deg + 1) @ coef)


def structural_halo_map(cube, center, q_per_px, halo_q, dq=0.06, flank_gap=0.12,
                        flank_w=0.05, vacuum_mask=None, q_max=1.2, nbin=200,
                        beam_cut=0.14, n_jobs=1, _stack=None):
    """Structural amorphous-halo map: the FSDP **bump above the smooth background**.

    A plain annular detector rises with total scattering (thickness), and even a
    steep but *featureless* background has intensity at the halo radius — so
    neither tells a genuine amorphous halo from mere thickness/background. Here a
    **straight line is fit to the two flank windows** (outside the halo, at
    ``halo_q ± flank_gap`` with width ``flank_w``) and evaluated under the band
    ``[halo_q - dq, halo_q + dq]``: the halo *bump above that local sloped trend*.
    A smooth (even steep) background lies on the line → residual ~0; only a real
    peak rises above it. Vacuum-referenced (:func:`significance`).

    **Beam guard.** For a *low-q* FSDP the symmetric low flank ``halo_q -
    flank_gap - flank_w`` can fall onto the bright, convex beam tail; anchoring
    the baseline there ruins the fit and zeroes the bump everywhere. So the low
    flank is clamped to start no lower than ``beam_cut``: if it would dip below,
    it is placed at ``[beam_cut, beam_cut + flank_w]`` (a low anchor just above
    the beam). This keeps the local straight-line baseline valid for a halo close
    to the beam without hand-tuning ``flank_gap``.

    Returns ``(sig, raw)`` over the scan grid.
    """
    scan = cube.scan_shape
    if _stack is None:
        q, prof = radial_stack(cube, center, q_per_px, q_max=q_max, nbin=nbin, n_jobs=n_jobs)
    else:
        q, prof = _stack
    band = (q >= halo_q - dq) & (q <= halo_q + dq)
    lo_start = halo_q - flank_gap - flank_w
    if lo_start < beam_cut:                       # low flank would sit in the beam
        lo_start, lo_end = beam_cut, beam_cut + flank_w
    else:
        lo_end = halo_q - flank_gap
    fl = ((q >= lo_start) & (q <= lo_end)) | \
         ((q >= halo_q + flank_gap) & (q <= halo_q + flank_gap + flank_w))
    if band.sum() == 0 or fl.sum() < 2:
        raw = np.zeros(prof.shape[0])
    else:
        qf = q[fl]
        # per-pixel linear fit over the flank points: slope, intercept
        qfm = qf.mean()
        dq_ = qf - qfm
        denom = float((dq_ ** 2).sum()) or 1.0
        slope = (prof[:, fl] * dq_).sum(1) / denom
        intercept = prof[:, fl].mean(1) - slope * qfm
        qc = q[band]
        base_center = slope[:, None] * qc[None, :] + intercept[:, None]   # (Npix, nband)
        raw = np.clip(prof[:, band] - base_center, 0.0, None).mean(1)
    raw = raw.reshape(scan)
    if vacuum_mask is None:
        vacuum_mask = strict_vacuum_mask(cube, center=center, q_per_px=q_per_px)
    return significance(raw, vacuum_mask), raw


def ring_phase_evidence(cube, center, q_per_px, candidates=None, ref_mask=None,
                        dq=0.03, flank=2.0, ring_sigma=4.0, q_lo=0.25, q_max=1.15,
                        nbin=200, n_jobs=1, _stack=None):
    """Per-phase polycrystalline-ring evidence, per scan pixel.

    For each candidate, a **flank-subtracted** ring detector is placed at every
    one of its crystalline ring d-spacings — a *sharp* polycrystalline ring shows
    above the broad amorphous halo, which (being broad) subtracts to ~0.

    The key reference for rings is the **amorphous material background**, not the
    vacuum: even amorphous material differs enormously from the near-silent vacuum,
    so a vacuum-referenced ring detector fires everywhere material sits. Instead
    each ring's flank-subtracted height is z-scored against its distribution over
    ``ref_mask`` (pass the material mask): the amorphous majority sets the median,
    and only a genuine sharp ring rises ``ring_sigma`` above it. A ring "counts"
    where it clears that; the phase's **evidence** is the strength-weighted fraction
    of its rings that count — several of a phase's rings lighting up together is
    what a single shared ring cannot fake. Only rings in ``[q_lo, q_max]`` are used.

    Returns ``{phase: dict(evidence, n_sig, rings)}`` — ``evidence`` and ``n_sig``
    are scan maps, ``rings`` is a list of ``(d, sig_map)``.
    """
    from .phases import COMPOUND_RINGS
    from .virtual_image import _resolve_center
    center = _resolve_center(cube, center)
    if q_per_px is None:
        q_per_px = cube.calibration.q_per_px
    names = list(candidates) if candidates is not None else list(COMPOUND_RINGS.keys())
    scan = cube.scan_shape
    if _stack is None:
        q, prof = radial_stack(cube, center, q_per_px, q_max=q_max + 0.05, nbin=nbin, n_jobs=n_jobs)
    else:
        q, prof = _stack
    if ref_mask is None:                                    # fall back: everything but strict vacuum
        ref_mask = ~strict_vacuum_mask(cube, center=center, q_per_px=q_per_px)
    ref_mask = np.asarray(ref_mask, bool)
    out = {}
    for c in names:
        rings = [(d, w) for d, w in COMPOUND_RINGS[c] if q_lo <= 1.0 / d <= q_max]
        wsum = sum(w for _, w in rings) + 1e-12
        evid = np.zeros(scan)
        nsig = np.zeros(scan, int)
        ring_sigs = []
        for d, w in rings:
            raw = peak_above_flank(q, prof, 1.0 / d, dq, flank=flank).reshape(scan)
            sig = significance(raw, ref_mask)               # z-score vs amorphous background
            hit = (sig > ring_sigma) & ref_mask
            evid += w * hit
            nsig += hit.astype(int)
            ring_sigs.append((d, sig))
        out[c] = dict(evidence=evid / wsum, n_sig=nsig, rings=ring_sigs)
    return out


def amorphous_halo_peaks(q, I, q_lo=0.15, q_hi=1.1, smooth=9, prominence_frac=0.02,
                         max_peaks=5, edge=0.03, min_width_q=0.0):
    """Broad amorphous maxima (FSDP + secondary halos) in a radial profile.

    Rolling-min baseline subtraction then peak picking on the smoothed residual —
    the diffuse halo bumps (e.g. ~0.2, 0.3, 0.4, 0.8 1/A) whose radii seed the
    halo detectors. Returns a list of ``q`` positions (1/A), strongest first.

    Peaks within ``edge`` of ``q_lo``/``q_hi`` are dropped: the rolling-min
    baseline cannot fully remove the steep beam tail right at the low boundary,
    so a spurious maximum otherwise appears pinned to ``q_lo`` (a beam-edge
    artifact, not a halo). ``find_peaks`` runs on the residual itself (not
    ``res * mask``, which would manufacture a step at the boundary).

    ``min_width_q`` > 0 keeps only peaks whose half-prominence width (in 1/A)
    exceeds it — an **amorphous halo is broad**, so this rejects a *sharp*
    crystalline ring that also produces a maximum in the (patch-averaged) mean
    profile. Use a small internal ``smooth`` when width-filtering so a sharp ring
    is not artificially broadened into the halo range.
    """
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    from scipy.signal import find_peaks, peak_widths
    q = np.asarray(q, float)
    I = np.asarray(I, float)
    w = max(5, int(len(q) * 0.12) | 1)
    base = uniform_filter1d(minimum_filter1d(I, w), w)
    res = uniform_filter1d(np.clip(I - base, 0, None), max(1, smooth))
    inwin = (q >= q_lo) & (q <= q_hi)
    if not inwin.any() or res[inwin].max() <= 0:
        return []
    keep = (q >= q_lo + edge) & (q <= q_hi - edge)
    pk, _ = find_peaks(res, prominence=prominence_frac * res[inwin].max())
    pk = pk[keep[pk]]
    if pk.size == 0:
        return []
    if min_width_q > 0:
        dqb = float(np.median(np.diff(q)))
        wq = peak_widths(res, pk, rel_height=0.5)[0] * dqb
        pk = pk[wq >= min_width_q]
        if pk.size == 0:
            return []
    order = np.argsort(-res[pk])
    return [float(q[i]) for i in pk[order][:max_peaks]]
