"""
fourdstem.analysis.classify
===========================
Per-pixel phase classification: run the halo / ring / spot analysis at *every*
scan position and assign each one a material and a confidence tier, then map it.

The flow the maps encode, per scan pixel:

  * **halo** (amorphous first-sharp-diffraction-peak) — present almost everywhere
    material sits; its strength says "there is (amorphous) material here".
  * **ring** (polycrystalline) — a sharp peak in the pixel's radial profile beyond
    the halo; matched against each candidate's ring fingerprint gives a per-phase
    ring score.
  * **spot** (single crystal) — discrete Bragg spots (high azimuthal variance);
    where a pixel's spots INDEX to one lattice at one zone axis it is confirmed.

Tier per pixel:
  * **확정 / confirmed** — the pixel's spots index to a phase (lattice self-consistent).
  * **예상 / predicted** — its ring fingerprint matches one phase (above threshold,
    winning clearly) but it does not index.
  * **약함 / weak** — material but no clear phase (halo only / ambiguous).

:func:`classify_pixels` returns the maps and representative example pixels for
the halo / ring / spot cases so the notebook can show both the maps and *why*.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import os
import numpy as np

from .phases import phase_ring_profile, CANDIDATES
from .indexing import crystallinity_map, seed_positions, index_seeds


def radial_stack(cube, center, q_per_px, q_max=1.2, nbin=180, chunk=512, n_jobs=1):
    """Radial profile ``I(q)`` for **every** scan position (vectorized).

    Returns ``(q, profiles)`` with ``q`` shape ``(nbin,)`` and ``profiles`` shape
    ``(Npix, nbin)`` (Npix = scan positions, row order = flattened scan). Uses a
    fixed detector->bin averaging matrix so the whole cube reduces with a few
    chunked mat-muls.
    """
    dp = cube.dp_shape
    H, W = dp
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.hypot(xx - center[0], yy - center[1]).ravel()
    qpx = r * q_per_px
    det = np.where(qpx <= q_max)[0]
    b = np.clip((qpx[det] / q_max * nbin).astype(int), 0, nbin - 1)
    # sort detector pixels by radial bin so each bin is a contiguous run -> segment
    # sums via reduceat (O(ndet) per pattern, not O(ndet*nbin))
    srt = np.argsort(b, kind="stable")
    det_s, b_s = det[srt], b[srt]
    present, first = np.unique(b_s, return_index=True)          # bins that occur, run starts
    counts = np.bincount(b_s, minlength=nbin).astype(np.float32)
    counts[counts == 0] = 1.0
    flat = cube._flat_patterns()
    N = flat.shape[0]
    flat2 = flat.reshape(N, -1)
    prof = np.zeros((N, nbin), np.float32)

    def _do(s):
        blk = np.asarray(flat2[s:s + chunk][:, det_s], np.float32)
        seg = np.add.reduceat(blk, first, axis=1)               # sum per occurring bin
        prof[s:s + blk.shape[0], present] = seg / counts[present]

    starts = list(range(0, N, chunk))
    from .indexing import _resolve_jobs
    nj = _resolve_jobs(n_jobs)
    if nj > 1 and len(starts) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=nj) as ex:
            list(ex.map(_do, starts))
    else:
        for s in starts:
            _do(s)
    q = (np.arange(nbin) + 0.5) / nbin * q_max
    return q, prof


@dataclass
class PixelClassification:
    scan: tuple
    q: np.ndarray
    candidates: list
    material: np.ndarray      # (Sy,Sx) bool, True = material (from the halo)
    halo: np.ndarray          # (Sy,Sx) amorphous strength
    ring: np.ndarray          # (Sy,Sx) sharp-ring strength
    spot: np.ndarray          # (Sy,Sx) azimuthal spottiness
    phase_idx: np.ndarray     # (Sy,Sx) best ring-fingerprint phase index (-1 none)
    phase_score: np.ndarray   # (Sy,Sx) best ring-fingerprint score
    tier: np.ndarray          # (Sy,Sx) 0 none,1 weak,2 predicted,3 confirmed
    indexed: list = field(default_factory=list)   # per-seed indexing results
    examples: dict = field(default_factory=dict)  # 'halo'/'ring'/'spot' -> (iy,ix)

    TIER = {0: "none", 1: "weak", 2: "predicted", 3: "confirmed"}

    def phase_name(self, iy, ix):
        k = self.phase_idx[iy, ix]
        return self.candidates[k] if k >= 0 else None

    def tier_map(self, tier):
        """Boolean map of positions at a given tier (1 weak, 2 predicted, 3 confirmed)."""
        return self.tier == tier

    def phase_tier_map(self, phase, tier):
        """Boolean map where best phase == ``phase`` AND tier == ``tier``."""
        k = self.candidates.index(phase)
        return (self.phase_idx == k) & (self.tier == tier)


def classify_pixels(cube, center=None, q_per_px=None, candidates=None, material=None,
                    halo_q=None, sigma_q=0.04, q_lo=0.30, q_max=1.15, nbin=180,
                    bg_win_frac=0.12, ring_thr=0.25, margin=1.15, spot_pctl=95.0,
                    index=True, spot_kwargs=None, index_kwargs=None, n_jobs=1):
    """Classify every scan pixel by halo / ring / spot and assign material + tier.

    Steps: per-pixel radial ``I(q)`` (:func:`radial_stack`); a rolling-min baseline
    gives the amorphous ``halo`` and the sharp ``ring`` residual; each candidate's
    ring fingerprint is matched (normalized dot with the residual over ``q_lo``..
    ``q_max``) into a per-phase score, whose argmax is the pixel's ring phase when
    it clears ``ring_thr`` and beats the runner-up by ``margin``; the azimuthal
    ``spot`` map (crystallinity) flags single-crystal pixels; and, if ``index``,
    spot pixels (above ``spot_pctl``) are seeded and indexed so a lattice-consistent
    pixel becomes **confirmed**. Returns a :class:`PixelClassification`.
    """
    from ..preprocess.masks import annular_mask
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    from .virtual_image import _resolve_center

    center = _resolve_center(cube, center)
    if q_per_px is None:
        q_per_px = cube.calibration.q_per_px
    names = list(candidates) if candidates is not None else list(CANDIDATES)
    scan = cube.scan_shape
    Sy, Sx = scan

    from .indexing import _resolve_jobs
    nj = _resolve_jobs(n_jobs)
    q, prof = radial_stack(cube, center, q_per_px, q_max=q_max + 0.05, nbin=nbin, n_jobs=nj)
    win = max(5, int(nbin * bg_win_frac) | 1)

    def _baseline(a):
        return uniform_filter1d(minimum_filter1d(a, win, axis=1), win, axis=1)

    if nj > 1 and prof.shape[0] > 2 * nj:              # thread the 1-D filters over row chunks
        from concurrent.futures import ThreadPoolExecutor
        rows = np.array_split(np.arange(prof.shape[0]), nj)
        base = np.empty_like(prof)
        with ThreadPoolExecutor(max_workers=nj) as ex:
            for idx, b in zip(rows, ex.map(lambda r: _baseline(prof[r]), rows)):
                base[idx] = b
    else:
        base = _baseline(prof)
    peaks = np.clip(prof - base, 0.0, None)

    sel = (q >= q_lo) & (q <= q_max)
    R = np.stack([phase_ring_profile(q[sel], c, sigma_q) for c in names], 1)   # (nqsel, nphase)
    R /= (np.linalg.norm(R, axis=0, keepdims=True) + 1e-9)
    Pk = peaks[:, sel]
    pn = np.linalg.norm(Pk, axis=1, keepdims=True) + 1e-9
    scores = (Pk / pn) @ R                                    # (Npix, nphase) cosine-like
    order = np.argsort(-scores, axis=1)
    best = order[:, 0]
    best_score = scores[np.arange(scores.shape[0]), best]
    second = scores[np.arange(scores.shape[0]), order[:, 1]]

    halo = base[:, sel].max(1)                                # amorphous baseline height
    ring = Pk.max(1)                                          # sharp ring residual height
    halo2d = halo.reshape(scan); ring2d = ring.reshape(scan)
    bscore2d = best_score.reshape(scan)
    bphase2d = best.reshape(scan)

    spot2d = np.asarray(crystallinity_map(cube, center=center, q_per_px=q_per_px,
                                          n_jobs=n_jobs), float)

    if material is None:                                      # derive material from the halo
        from .virtual_image import _otsu_threshold
        thr = _otsu_threshold(halo)
        mat = halo2d > thr
        if mat.mean() < 0.02 or mat.mean() > 0.98:           # degenerate -> fall back to a percentile
            mat = halo2d > np.percentile(halo, 60)
    else:
        mat = np.asarray(material, bool)
    # ring phase accepted where score clears threshold and beats runner-up by margin
    ring_ok = mat & (best_score.reshape(scan) >= ring_thr) & \
        (best_score >= margin * second).reshape(scan)
    phase_idx = np.where(ring_ok, bphase2d, -1)

    tier = np.zeros(scan, int)
    tier[mat] = 1                                            # material w/o clear phase = weak
    tier[ring_ok] = 2                                        # ring fingerprint match = predicted

    indexed = []
    if index:
        spot_kwargs = spot_kwargs or {}
        index_kwargs = index_kwargs or {}
        seeds = seed_positions([spot2d], mask=mat, threshold_pctl=spot_pctl)
        indexed = index_seeds(cube, seeds, center=center, q_per_px=q_per_px,
                              candidates=names, mask=mat,
                              spot_kwargs=spot_kwargs, index_kwargs=index_kwargs,
                              n_jobs=n_jobs)
        for r in indexed:
            b = r["best"]
            if b is not None and b.get("indexed"):
                iy, ix = r["pos"]
                tier[iy, ix] = 3
                phase_idx[iy, ix] = names.index(b["phase"])

    # representative example pixels (highest of each signal type, within material)
    def _argmax_in(a):
        am = np.where(mat, a, -np.inf)
        return tuple(int(v) for v in np.unravel_index(np.argmax(am), scan))
    examples = dict(halo=_argmax_in(halo2d), ring=_argmax_in(ring2d), spot=_argmax_in(spot2d))

    return PixelClassification(scan=scan, q=q, candidates=names, material=mat,
                               halo=halo2d, ring=ring2d, spot=spot2d,
                               phase_idx=phase_idx, phase_score=bscore2d,
                               tier=tier, indexed=indexed, examples=examples)
