"""
fourdstem.preprocess.center
===========================
Beam-center estimation for diffraction patterns.

Two complementary methods:

* ``center_of_mass`` — fast, works well when a strong central beam dominates
  (typical for a bright-field disk). Optionally thresholded to the bright core.
* ``find_center_friedel`` — robust for amorphous rings AND spotty crystalline
  patterns: it maximizes the Friedel symmetry correlation I(r,θ) ≈ I(r,θ+π)
  about a trial center. This is the method carried over from the RDF pipeline.

``find_center`` picks a sensible default (Friedel) and seeds it from the
center-of-mass estimate.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize


def center_of_mass(img, mask=None, threshold=None):
    """Intensity-weighted centroid ``(cx, cy)`` of ``img``.

    ``threshold`` (fraction of max) restricts the CoM to the bright core, which
    is more stable when there is a diffuse background.
    """
    img = np.asarray(img, float)
    w = img.copy()
    if mask is not None:
        w = np.where(mask, 0.0, w)
    if threshold is not None:
        w = np.where(w >= threshold * np.nanmax(w), w, 0.0)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    tot = w.sum()
    if tot <= 0:
        H, W = img.shape
        return (W / 2.0, H / 2.0)
    H, W = img.shape
    yy, xx = np.mgrid[0:H, 0:W]
    return (float((xx * w).sum() / tot), float((yy * w).sum() / tot))


def _sample_ring(img, mask, cx, cy, R, TH):
    X = cx + R * np.cos(TH)
    Y = cy + R * np.sin(TH)
    v = map_coordinates(img, [Y.ravel(), X.ravel()], order=1,
                        mode="constant", cval=np.nan).reshape(TH.shape)
    if mask is not None:
        mm = map_coordinates(mask.astype(float), [Y.ravel(), X.ravel()],
                             order=0, mode="constant", cval=1).reshape(TH.shape)
        v[mm > 0.5] = np.nan
    return v


def friedel_correlation(img, cx, cy, r0, r1, mask=None, n_r=220, n_th=360):
    """Normalized correlation between I(r,θ) and I(r,θ+π) about ``(cx, cy)``.

    Returns a value in [-1, 1]; 1 means perfect centro-symmetry. Returns -1 if
    too few valid samples fall in the annulus.
    """
    rs = np.linspace(r0, r1, n_r)
    R, TH = np.meshgrid(rs, np.linspace(0, np.pi, n_th, endpoint=False))
    a = _sample_ring(img, mask, cx, cy, R, TH)
    b = _sample_ring(img, mask, cx, cy, R, TH + np.pi)
    g = np.isfinite(a) & np.isfinite(b)
    a, b = a[g], b[g]
    if a.size < 50:
        return -1.0
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9
    return float((a * b).sum() / denom)


def find_center_friedel(img, mask=None, r_frac=(0.06, 0.46), seed=None,
                        maxiter=400):
    """Refine the beam center by maximizing Friedel symmetry.

    Returns ``((cx, cy), correlation)``.
    """
    img = np.asarray(img, float)
    H, W = img.shape
    r0, r1 = r_frac[0] * min(H, W), r_frac[1] * min(H, W)
    if seed is None:
        bright = img > 0.6 * np.nanmax(img)
        ys, xs = np.where(bright)
        seed = (xs.mean(), ys.mean()) if xs.size else (W / 2.0, H / 2.0)
    res = minimize(
        lambda c: -friedel_correlation(img, c[0], c[1], r0, r1, mask),
        np.array(seed, float), method="Nelder-Mead",
        options=dict(xatol=0.1, fatol=1e-5, maxiter=maxiter),
    )
    cx, cy = res.x
    return (float(cx), float(cy)), friedel_correlation(img, cx, cy, r0, r1, mask)


def find_center(img, mask=None, method="friedel"):
    """Convenience dispatcher. ``method`` in {"friedel", "com"}.

    Returns ``((cx, cy), quality)`` where quality is the Friedel correlation
    (friedel) or ``nan`` (com).
    """
    if method == "com":
        return center_of_mass(img, mask, threshold=0.5), float("nan")
    seed = center_of_mass(img, mask, threshold=0.5)
    return find_center_friedel(img, mask, seed=seed)
