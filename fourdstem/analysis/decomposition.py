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


def _row_normalize(X, mode):
    """Per-pattern (row) normalization to remove per-position intensity drift.

    In a 4D-STEM scan the total counts per position vary (dose, thickness,
    beam-current / scan-line drift). Without normalization a decomposition
    separates patterns by BRIGHTNESS (often showing up as scan-line stripes in
    the loading maps) instead of by STRUCTURE. Dividing each pattern by its
    integrated intensity makes the factorization compare shapes.
    """
    if mode in (None, False):
        return X
    if mode in ("sum", True):
        s = X.sum(axis=1, keepdims=True)
    elif mode == "mean":
        s = X.mean(axis=1, keepdims=True)
    elif mode == "l2":
        s = np.sqrt((X * X).sum(axis=1, keepdims=True))
    else:
        raise ValueError(f"unknown normalize mode {mode!r}")
    return X / np.where(s > 0, s, 1.0)


def nmf_decompose(cube, n_components=4, mask=None, normalize=None, max_iter=400,
                  init="nndsvda", random_state=0, **kwargs):
    """Non-negative matrix factorization of a cube.

    Parameters
    ----------
    cube : DataCube or ndarray
    n_components : int
        Number of components ``k``.
    mask : 2D bool array, optional
        Detector pixels to exclude (``True`` = drop), e.g. a beam-stop + central
        beam mask. Excluded pixels also don't count toward ``normalize``.
    normalize : {None, "sum", "mean", "l2"}
        Per-pattern normalization BEFORE factorization. Use ``"sum"`` to separate
        by structure rather than brightness (removes scan-line intensity drift).
        Pair with a mask that hides the central beam so the normalization
        reflects the structural (scattered) intensity.
    max_iter, init, random_state, **kwargs
        Passed to :class:`sklearn.decomposition.NMF`.

    Returns
    -------
    DecompositionResult
    """
    from sklearn.decomposition import NMF

    X, dp_shape, scan_shape, keep = _unfold(cube, mask)
    X = np.clip(X, 0, None)  # NMF requires non-negativity
    X = _row_normalize(X, normalize)
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


def pca_decompose(cube, n_components=8, mask=None, normalize=None, whiten=False,
                  random_state=0, **kwargs):
    """Principal component analysis of a cube (denoising / rank estimation).

    Returns a :class:`DecompositionResult` whose ``components`` are the principal
    diffraction patterns and ``loadings`` the score maps. Mean-centering is done
    by scikit-learn internally; the subtracted mean is available on
    ``result.model.mean_``. ``normalize`` (see :func:`nmf_decompose`) divides
    each pattern by its intensity first, so PCA finds structural (not brightness)
    variation.
    """
    from sklearn.decomposition import PCA

    X, dp_shape, scan_shape, keep = _unfold(cube, mask)
    X = _row_normalize(X, normalize)
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


def cluster_cube(cube, n_clusters=2, center=None, mask=None, feature="radial",
                 n_bins=None, normalize="sum", random_state=0):
    """k-means cluster scan positions into phases (hard assignment, no threshold).

    Each scan position is assigned to exactly one cluster from its diffraction
    *pattern*, so region boundaries follow the data instead of a scalar cutoff
    that bleeds across boundaries. Returns ``(label_map, cluster_patterns, model)``:

    * ``label_map`` — cluster index at each scan position (scan-shaped)
    * ``cluster_patterns`` — ``(n_clusters, Qy, Qx)`` mean diffraction pattern of
      each cluster (feed to ``pattern_to_rdf`` for a per-phase RDF)
    * ``model`` — the fitted KMeans

    ``feature="radial"`` clusters on per-position I(q) (denoised, recommended);
    ``"pixel"`` clusters on the raw masked pixels. ``normalize`` removes
    per-position brightness so clusters reflect structure.
    """
    from sklearn.cluster import KMeans

    if center is None:
        from ..preprocess.center import find_center
        from ..preprocess.masks import beam_stopper_mask
        md = cube.mean_dp()
        center, _ = find_center(md, beam_stopper_mask(md))

    if feature == "radial":
        from .azimuthal import radial_profiles
        X, _ = radial_profiles(cube, center, n_bins=n_bins, mask=mask,
                               normalize=(normalize is not None))
    else:
        X, dp_shape, scan_shape, keep = _unfold(cube, mask)
        X = _row_normalize(np.clip(X, 0, None), normalize)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(np.nan_to_num(X, nan=0.0))

    scan_shape = cube.scan_shape
    label_map = labels.reshape(scan_shape) if scan_shape else labels
    flat = cube._flat_patterns()
    patterns = np.stack([flat[labels == k].mean(0) if np.any(labels == k)
                         else np.zeros(cube.dp_shape) for k in range(n_clusters)])
    return label_map, patterns, km


@dataclass
class ProfileDecomposition:
    """Decomposition of a stack of 1D profiles (e.g. G(r) vs temperature).

    Attributes
    ----------
    components : ndarray, shape (k, n_features)
        Basis profiles (rows of H). ``components[i]`` is a characteristic curve.
    weights : ndarray, shape (n_samples, k)
        How much of each component each sample contains (rows of W).
    fractions : ndarray, shape (n_samples, k)
        ``weights`` normalized so each sample's row sums to 1 — the "NMF fraction"
        (e.g. amorphous vs crystalline abundance at each temperature).
    x : ndarray or None
        Shared feature axis (e.g. the r grid), if provided.
    method : str
    model : object
    explained_variance_ratio : ndarray or None
    """
    components: np.ndarray
    weights: np.ndarray
    fractions: np.ndarray
    x: np.ndarray | None = None
    method: str = "nmf"
    model: object = None
    explained_variance_ratio: np.ndarray | None = None


def decompose_profiles(profiles, n_components=2, method="nmf", x=None,
                       normalize_fraction=True, random_state=0, **kwargs):
    """Decompose a stack of 1D profiles into ``k`` components + per-sample weights.

    This is the workhorse for an in-situ series: stack every temperature's G(r)
    (or I(q), φ(q)) into a matrix ``X`` of shape ``(n_samples, n_features)`` and
    factor ``X ≈ W · H``. With NMF and ``k=2`` you typically resolve two physical
    end-members and get their fraction at each temperature.

    Parameters
    ----------
    profiles : array (n_samples, n_features) or list of 1D arrays
        The stacked profiles (must share length/feature axis).
    n_components : int
        Number of components ``k`` (start with 2; increase as needed).
    method : {"nmf", "pca"}
    x : array, optional
        The shared feature axis (e.g. r grid) stored on the result for plotting.
    normalize_fraction : bool
        Compute ``fractions`` = weights normalized per sample to sum to 1.
    random_state : int
    **kwargs
        Passed to the scikit-learn estimator.

    Returns
    -------
    ProfileDecomposition
    """
    X = np.asarray(profiles, float)
    if X.ndim != 2:
        X = np.vstack([np.asarray(p, float).ravel() for p in profiles])

    if method == "nmf":
        from sklearn.decomposition import NMF
        # NMF requires non-negative data. A reduced PDF G(r) (or phi(q)) is
        # SIGNED — its troughs/baseline are negative — so clipping them to 0
        # destroys real structure and yields meaningless components. Warn loudly;
        # for signed profiles use method="pca" instead.
        neg_frac = float(np.mean(X < -1e-9))
        if neg_frac > 0.02:
            import warnings
            warnings.warn(
                f"decompose_profiles(method='nmf') got data that is {neg_frac:.0%} "
                "negative (e.g. G(r)/phi(q) have troughs). NMF needs non-negative "
                "input, so negatives are clipped to 0 — components will be "
                "distorted. Use method='pca' for signed profiles; keep NMF for "
                "non-negative data like diffraction patterns / I(q)."
            )
        Xn = np.clip(np.nan_to_num(X, nan=0.0), 0, None)
        model = NMF(n_components=n_components, init="nndsvda",
                    random_state=random_state,
                    max_iter=kwargs.pop("max_iter", 500), **kwargs)
        W = model.fit_transform(Xn)
        H = model.components_
        evr = None
    elif method == "pca":
        from sklearn.decomposition import PCA
        model = PCA(n_components=n_components, random_state=random_state, **kwargs)
        W = model.fit_transform(np.nan_to_num(X, nan=0.0))
        H = model.components_
        evr = model.explained_variance_ratio_
    else:
        raise ValueError(f"unknown method {method!r}")

    if normalize_fraction:
        wsum = W.sum(axis=1, keepdims=True)
        fractions = W / np.where(wsum == 0, 1.0, wsum)
    else:
        fractions = W.copy()

    return ProfileDecomposition(
        components=H, weights=W, fractions=fractions,
        x=None if x is None else np.asarray(x, float),
        method=method, model=model, explained_variance_ratio=evr,
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
