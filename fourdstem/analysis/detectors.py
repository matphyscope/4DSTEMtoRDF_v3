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
    """Annular peak height at ``q0`` above the local background, per position.

    Mean intensity in the detector band ``[q0-dq, q0+dq]`` minus the mean over the
    two flanking bands (``dq..flank*dq`` on each side). A real halo/ring peak gives
    a positive value; a smooth (thickness) background subtracts to ~0, so this is
    thickness-insensitive and isolates *structure*. ``prof`` is ``(Npix, nbin)``.
    """
    band = (q >= q0 - dq) & (q <= q0 + dq)
    lo = (q >= q0 - flank * dq) & (q < q0 - dq)
    hi = (q > q0 + dq) & (q <= q0 + flank * dq)
    if band.sum() == 0:
        return np.zeros(prof.shape[0])
    center = prof[:, band].mean(1)
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


def amorphous_halo_peaks(q, I, q_lo=0.15, q_hi=1.1, smooth=9, prominence_frac=0.02,
                         max_peaks=5):
    """Broad amorphous maxima (FSDP + secondary halos) in a radial profile.

    Rolling-min baseline subtraction then peak picking on the smoothed residual —
    the diffuse halo bumps (e.g. ~0.2, 0.3, 0.4, 0.8 1/A) whose radii seed the
    halo detectors. Returns a list of ``q`` positions (1/A), strongest first.
    """
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    from scipy.signal import find_peaks
    q = np.asarray(q, float)
    I = np.asarray(I, float)
    w = max(5, int(len(q) * 0.12) | 1)
    base = uniform_filter1d(minimum_filter1d(I, w), w)
    res = uniform_filter1d(np.clip(I - base, 0, None), max(3, smooth))
    m = (q >= q_lo) & (q <= q_hi)
    if not m.any() or res[m].max() <= 0:
        return []
    pk, props = find_peaks(res * m, prominence=prominence_frac * res[m].max())
    order = np.argsort(-res[pk])
    return [float(q[i]) for i in pk[order][:max_peaks]]
