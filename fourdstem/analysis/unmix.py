"""
fourdstem.analysis.unmix
========================
Supervised phase identification by non-negative unmixing against KNOWN
compound signatures — the opposite of blind NMF. Instead of letting the data
invent components, we build a reference G(r) for each candidate compound from
its crystallographic shell distances and ask, at every probe (or per phase),
"how much of each compound explains this RDF?" via non-negative least squares:

    G_measured(r) ≈ Σ_c  a_c · G_ref^c(r),      a_c ≥ 0

Compounds that are absent simply get a_c ≈ 0, so passing five candidates when
only two are present is fine.

Caveat (resolution): at low q_max the shells broaden (Δr ≈ 1/q_max). Compounds
whose first shells nearly coincide (LiF/Li2O/Li3N all ~2.0 Å) become collinear
references — NNLS then splits them somewhat arbitrarily. Use
:func:`reference_degeneracy` to see which references are separable at your
resolution; Li2S (2.48 Å) and Li2CO3 (short C–O 1.28 Å) stay distinct.
"""
from __future__ import annotations
import numpy as np


# Shell model per compound: (distance Å, coordination number, Z_i, Z_j).
# Crystallographic nearest distances; amplitude ~ CN·Z_i·Z_j (electron RDF weight).
# Z: Li 3, C 6, N 7, O 8, F 9, S 16.
COMPOUND_SHELLS = {
    "LiF":    [(2.013, 6, 3, 9), (2.847, 12, 9, 9), (2.847, 12, 3, 3), (3.486, 8, 3, 9)],
    "Li2O":   [(2.00, 4, 3, 8), (2.31, 12, 3, 3), (3.267, 12, 8, 8)],
    "Li3N":   [(1.938, 2, 3, 7), (2.11, 6, 3, 7), (3.65, 6, 7, 7)],
    "Li2CO3": [(1.28, 3, 6, 8), (2.00, 4, 3, 8), (2.22, 2, 8, 8)],
    "Li2S":   [(2.475, 4, 3, 16), (2.858, 12, 3, 3), (4.042, 12, 16, 16)],
}


def synth_compound_rdf(shells, r, sigma=0.5, weight="ZZ"):
    """Synthetic G(r) for one compound: sum of Gaussian shells.

    ``shells`` is a list of ``(distance, CN, Z_i, Z_j)``. ``sigma`` (Å) sets the
    broadening — match it to the data resolution (≈ 0.5/q_max). ``weight="ZZ"``
    scales each shell by CN·Z_i·Z_j (electron scattering); ``"cn"`` uses the
    coordination number only; ``"flat"`` weights all shells equally.
    """
    r = np.asarray(r, float)
    g = np.zeros_like(r)
    for d, cn, zi, zj in shells:
        if weight == "ZZ":
            w = cn * zi * zj
        elif weight == "cn":
            w = cn
        else:
            w = 1.0
        g += w * np.exp(-0.5 * ((r - d) / sigma) ** 2)
    return g


def build_references(r, compounds=None, sigma=0.5, weight="ZZ", normalize=True):
    """Reference G(r) for each candidate compound on grid ``r`` -> ``dict``."""
    compounds = list(compounds) if compounds is not None else list(COMPOUND_SHELLS)
    refs = {}
    for c in compounds:
        if c not in COMPOUND_SHELLS:
            raise KeyError(f"no shell model for {c!r}; known: {list(COMPOUND_SHELLS)}")
        g = synth_compound_rdf(COMPOUND_SHELLS[c], r, sigma=sigma, weight=weight)
        if normalize:
            n = np.linalg.norm(g)
            if n > 0:
                g = g / n
        refs[c] = g
    return refs


def unmix_nnls(data, refs, r=None, r_range=None, clip_negative=True,
               normalize_fraction=True):
    """Non-negative unmixing of RDF(s) onto compound references.

    Parameters
    ----------
    data : (nr,) or (n, nr) array
        One RDF or a stack (per phase / per position).
    refs : dict name->(nr,) or (k, nr) array
        Compound references on the SAME r-grid as ``data``.
    r, r_range : optional
        If both given, restrict the fit to ``r_range=(lo, hi)`` (Å) — usually the
        peak region above the low-r straight line.
    clip_negative : bool
        Clip the measured G(r) at 0 before fitting (references are positive
        peaks; the RDF's negative lobes/ripples then contribute only to residual
        rather than dragging the fit).
    normalize_fraction : bool
        Scale each row's abundances to sum to 1 (fractions). Rows that fit to all
        zeros stay zero.

    Returns
    -------
    names : list[str]
    abundances : (k,) or (n, k) array
    residual : float or (n,) array   (relative NNLS residual, lower = better fit)
    """
    from scipy.optimize import nnls

    if isinstance(refs, dict):
        names = list(refs)
        R = np.vstack([np.asarray(refs[c], float) for c in names]).T   # (nr, k)
    else:
        R = np.asarray(refs, float).T
        names = [f"c{i}" for i in range(R.shape[1])]

    single = np.asarray(data).ndim == 1
    D = np.atleast_2d(np.asarray(data, float))
    if r is not None and r_range is not None:
        r = np.asarray(r, float)
        m = (r >= r_range[0]) & (r <= r_range[1])
        R = R[m]
        D = D[:, m]
    if clip_negative:
        D = np.clip(D, 0, None)

    A = np.zeros((D.shape[0], R.shape[1]))
    res = np.zeros(D.shape[0])
    for i, row in enumerate(D):
        norm = np.linalg.norm(row) + 1e-12
        a, rn = nnls(R, row)
        A[i] = a
        res[i] = rn / norm
    if normalize_fraction:
        s = A.sum(1, keepdims=True)
        A = np.divide(A, s, out=np.zeros_like(A), where=s > 0)
    if single:
        return names, A[0], float(res[0])
    return names, A, res


def reference_degeneracy(refs):
    """Pairwise cosine similarity of the references (1 = indistinguishable).

    Returns ``(names, C)`` where ``C[i,j]`` is the normalized dot product. High
    off-diagonal values flag compound pairs that this resolution cannot separate.
    """
    names = list(refs)
    R = np.vstack([np.asarray(refs[c], float) for c in names])
    Rn = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)
    return names, Rn @ Rn.T
