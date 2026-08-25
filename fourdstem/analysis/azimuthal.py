"""
fourdstem.analysis.azimuthal
===========================
Azimuthal (radial) integration of a diffraction pattern into a 1D profile
I(q), with an optional mask. Also provides the azimuthal *variance*, a useful
texture/anisotropy metric (high where sharp Bragg spots break ring symmetry).
"""
from __future__ import annotations
import numpy as np

from ..utils.helpers import radial_coordinate


def azimuthal_integrate(img, center, q_per_px, mask=None, q_grid=None,
                        statistic="mean"):
    """Radially integrate ``img`` about ``center`` into a 1D profile.

    Parameters
    ----------
    img : 2D array
    center : (cx, cy)
    q_per_px : float
        Å⁻¹ per pixel. Pass ``1.0`` to work in pixel units.
    mask : 2D bool array, optional
        ``True`` = exclude (beam stop, Bragg spots).
    q_grid : 1D array, optional
        Bin edges in q. Defaults to one bin per pixel-ring out to the largest
        fully-sampled q.
    statistic : {"mean", "sum", "median"}
        Reduction within each radial bin.

    Returns
    -------
    q_centers, I : 1D arrays
    """
    img = np.asarray(img, float)
    H, W = img.shape
    cx, cy = center
    q = radial_coordinate(img.shape, center) * q_per_px

    good = np.isfinite(img)
    if mask is not None:
        good &= ~np.asarray(mask, bool)

    if q_grid is None:
        q_edge = min(cx, cy, W - cx, H - cy) * q_per_px
        q_grid = np.arange(0.0, q_edge, q_per_px)

    qb = q[good].ravel()
    vb = img[good].ravel()
    idx = np.digitize(qb, q_grid)
    qc = 0.5 * (q_grid[:-1] + q_grid[1:])

    if statistic == "median":
        prof = np.full(len(q_grid) + 1, np.nan)
        for b in range(1, len(q_grid)):
            sel = vb[idx == b]
            if sel.size:
                prof[b] = np.median(sel)
        return qc, prof[1:len(q_grid)]

    cnt = np.bincount(idx, minlength=len(q_grid) + 1).astype(float)
    tot = np.bincount(idx, weights=vb, minlength=len(q_grid) + 1)
    if statistic == "sum":
        return qc, tot[1:len(q_grid)]
    with np.errstate(invalid="ignore", divide="ignore"):
        Iq = tot / cnt
    return qc, Iq[1:len(q_grid)]


def radial_profiles(cube, center, n_bins=None, mask=None, normalize=False,
                    chunk=2048):
    """Per-scan-position radial intensity I(q) — one profile per position.

    Returns ``(profiles, r_centers)`` where ``profiles`` has shape
    ``(n_position, n_bins)``. Azimuthal averaging denoises each pattern into a
    compact structural feature vector — ideal input for clustering (phase
    mapping) or per-position structural metrics, far more robust than raw pixels.
    ``normalize=True`` scales each profile to unit sum (compare shape, not dose).

    Memory: positions are processed in blocks of ``chunk`` and kept in float32,
    and the per-bin averaging is a single sparse matrix product, so peak memory
    is ``chunk × n_kept_pixels`` rather than a float64 copy of the whole cube
    (which used to crash the kernel on large scans).
    """
    from scipy import sparse

    dp = cube.dp_shape
    flat = cube._flat_patterns()
    n_pos = flat.shape[0]
    r = radial_coordinate(dp, center)
    cx, cy = center
    rmax = min(cx, cy, dp[1] - cx, dp[0] - cy)
    if n_bins is None:
        n_bins = int(max(8, rmax))
    edges = np.linspace(0, rmax, n_bins + 1)
    idx = np.digitize(r.ravel(), edges) - 1
    keep = (idx >= 0) & (idx < n_bins)
    if mask is not None:
        keep &= ~np.asarray(mask, bool).ravel()
    idx_k = idx[keep].astype(np.intp)
    counts = np.bincount(idx_k, minlength=n_bins).astype(np.float32)
    counts[counts == 0] = 1.0
    # sparse (n_kept x n_bins) averaging operator:  profiles = F_kept @ M
    M = sparse.csr_matrix(
        (1.0 / counts[idx_k], (np.arange(idx_k.size), idx_k)),
        shape=(idx_k.size, n_bins), dtype=np.float32)

    flat2 = flat.reshape(n_pos, -1)
    prof = np.zeros((n_pos, n_bins), np.float32)
    for s in range(0, n_pos, chunk):
        e = min(s + chunk, n_pos)
        Fk = np.asarray(flat2[s:e][:, keep], dtype=np.float32)   # one block only
        prof[s:e] = Fk @ M
    if normalize:
        ssum = prof.sum(1, keepdims=True)
        prof = prof / np.where(ssum > 0, ssum, 1.0)
    return prof, 0.5 * (edges[:-1] + edges[1:])


def azimuthal_variance(img, center, q_per_px, mask=None, q_grid=None):
    """Per-ring variance normalized by the ring mean squared: ``var(I)/mean(I)²``.

    Peaks in this profile flag q where the intensity is *anisotropic* around the
    ring — i.e. Bragg spots rather than a smooth amorphous halo. Returns
    ``(q_centers, V)``.
    """
    img = np.asarray(img, float)
    H, W = img.shape
    cx, cy = center
    q = radial_coordinate(img.shape, center) * q_per_px
    good = np.isfinite(img)
    if mask is not None:
        good &= ~np.asarray(mask, bool)
    if q_grid is None:
        q_edge = min(cx, cy, W - cx, H - cy) * q_per_px
        q_grid = np.arange(0.0, q_edge, q_per_px)
    qb = q[good].ravel()
    vb = img[good].ravel()
    idx = np.digitize(qb, q_grid)
    n = len(q_grid)
    cnt = np.bincount(idx, minlength=n + 1).astype(float)
    s1 = np.bincount(idx, weights=vb, minlength=n + 1)
    s2 = np.bincount(idx, weights=vb * vb, minlength=n + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / cnt
        var = s2 / cnt - mean ** 2
        V = var / (mean ** 2)
    qc = 0.5 * (q_grid[:-1] + q_grid[1:])
    return qc, V[1:n]
