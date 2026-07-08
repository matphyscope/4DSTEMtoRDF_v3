"""
fourdstem.analysis.decomposition
================================
Unsupervised decomposition of a 4D-STEM cube into a small set of diffraction
"components" and their real-space loading maps.

Given a cube of shape ``(R_Ny, R_Nx, Q_Ny, Q_Nx)`` we build the unfolded data
matrix ``X`` of shape ``(n_patterns, n_detector_pixels)`` and factor it:

    X ≈ W @ H         W: (n_patterns, k) loadings   H: (k, n_detector_pixels) components

* **NMF** (non-negative) is the natural choice for diffraction data, which is
  non-negative counts: components look like physical partial diffraction
  patterns and loadings are non-negative abundance maps. This is what you asked
  for — separating, say, an amorphous-halo component from a crystalline-Bragg
  component and mapping where each lives.
* **PCA** is offered for denoising / dimensionality estimation (scree plot).

For an in-situ *series*, stack the per-frame mean patterns into a 3D cube and
decompose that: components are recurring diffraction motifs and loadings are
their strength vs. frame (time/temperature).

Requires scikit-learn.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from ..io.datacube import DataCube


@dataclass
class DecompositionResult:
    """Result of a cube decomposition.

    Attributes
    ----------
    components : ndarray, shape (k, Q_Ny, Q_Nx)
        The component diffraction patterns (rows of H, refolded to 2D).
    loadings : ndarray
        Real-space loading maps, shape ``(k, *scan_shape)`` (or ``(k, n)`` for a
        3D stack). ``loadings[i]`` is the abundance of component ``i``.
    method : str
    explained_variance_ratio : ndarray or None
        For PCA only.
    model : object
        The fitted scikit-learn estimator (for reuse / inspection).
    """
    components: np.ndarray
    loadings: np.ndarray
    method: str
    explained_variance_ratio: np.ndarray | None = None
    model: object = None


def _unfold(cube, mask=None):
    """Return ``(X, dp_shape, scan_shape, keep_index)``.

    ``X`` has one row per pattern and one column per *kept* detector pixel.
    """
    if isinstance(cube, DataCube):
        flat = cube._flat_patterns().astype(float)
        dp_shape = cube.dp_shape
        scan_shape = cube.scan_shape
    else:
        arr = np.asarray(cube, float)
        dp_shape = arr.shape[-2:]
        scan_shape = arr.shape[:-2]
        flat = arr.reshape(-1, *dp_shape)
    n = flat.shape[0]
    X = flat.reshape(n, -1)
    if mask is not None:
        keep = ~np.asarray(mask, bool).ravel()
        X = X[:, keep]
    else:
        keep = None
    return X, dp_shape, scan_shape, keep


def _refold_components(H, dp_shape, keep):
    """Put component rows back onto the full 2D detector grid (masked -> 0)."""
    k = H.shape[0]
    full = np.zeros((k, int(np.prod(dp_shape))), float)
    if keep is None:
        full = H
    else:
        full[:, keep] = H
    return full.reshape(k, *dp_shape)


def _refold_loadings(W, scan_shape):
    k = W.shape[1]
    loads = W.T  # (k, n_patterns)
    if scan_shape:
        return loads.reshape(k, *scan_shape)
    return loads


def nmf_decompose(cube, n_components=4, mask=None, max_iter=400,
                  init="nndsvda", random_state=0, **kwargs):
    """Non-negative matrix factorization of a cube.

    Parameters
    ----------
    cube : DataCube or ndarray
    n_components : int
        Number of components ``k``.
    mask : 2D bool array, optional
        Detector pixels to exclude (``True`` = drop), e.g. a beam-stop mask.
    max_iter, init, random_state, **kwargs
        Passed to :class:`sklearn.decomposition.NMF`.

    Returns
    -------
    DecompositionResult
    """
    from sklearn.decomposition import NMF

    X, dp_shape, scan_shape, keep = _unfold(cube, mask)
    X = np.clip(X, 0, None)  # NMF requires non-negativity
    model = NMF(n_components=n_components, init=init, max_iter=max_iter,
                random_state=random_state, **kwargs)
    W = model.fit_transform(X)
    H = model.components_
    return DecompositionResult(
        components=_refold_components(H, dp_shape, keep),
        loadings=_refold_loadings(W, scan_shape),
        method="nmf",
        model=model,
    )


def pca_decompose(cube, n_components=8, mask=None, whiten=False,
                  random_state=0, **kwargs):
    """Principal component analysis of a cube (denoising / rank estimation).

    Returns a :class:`DecompositionResult` whose ``components`` are the principal
    diffraction patterns and ``loadings`` the score maps. Mean-centering is done
    by scikit-learn internally; the subtracted mean is available on
    ``result.model.mean_``.
    """
    from sklearn.decomposition import PCA

    X, dp_shape, scan_shape, keep = _unfold(cube, mask)
    model = PCA(n_components=n_components, whiten=whiten,
                random_state=random_state, **kwargs)
    W = model.fit_transform(X)
    H = model.components_
    return DecompositionResult(
        components=_refold_components(H, dp_shape, keep),
        loadings=_refold_loadings(W, scan_shape),
        method="pca",
        explained_variance_ratio=model.explained_variance_ratio_,
        model=model,
    )


def reconstruct(result: DecompositionResult, dp_shape=None):
    """Reconstruct the approximate data matrix ``W @ H`` from a result.

    Returns an array shaped ``(*scan_shape, Q_Ny, Q_Nx)``. Useful for denoising
    (keep a few PCA components) or residual inspection.
    """
    comps = result.components
    k = comps.shape[0]
    dp = comps.shape[1:]
    H = comps.reshape(k, -1)
    loads = result.loadings.reshape(k, -1).T  # (n_patterns, k)
    recon = loads @ H
    if result.method == "pca" and result.model is not None:
        recon = recon + result.model.mean_
    scan_shape = result.loadings.shape[1:]
    return recon.reshape(*scan_shape, *dp)
