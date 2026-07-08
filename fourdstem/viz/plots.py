"""
fourdstem.viz.plots
==================
Thin matplotlib helpers for the common figures. Every function accepts an
optional ``ax`` and returns it, so they compose into subplot grids. matplotlib
is imported lazily so the rest of the package works without it.
"""
from __future__ import annotations
import numpy as np


def _ax(ax):
    if ax is None:
        import matplotlib.pyplot as plt
        _, ax = plt.subplots()
    return ax


def show_pattern(img, center=None, mask=None, ax=None, log=True, cmap="magma",
                 title=None):
    """Display a diffraction pattern, optionally log-scaled with center + mask."""
    ax = _ax(ax)
    img = np.asarray(img, float)
    disp = np.log1p(np.clip(img - np.nanmin(img), 0, None)) if log else img
    ax.imshow(disp, cmap=cmap, origin="upper")
    if mask is not None:
        ax.imshow(np.ma.masked_where(~np.asarray(mask, bool), np.ones_like(img)),
                  cmap="cool", alpha=0.35, origin="upper")
    if center is not None:
        ax.plot(center[0], center[1], "c+", ms=12, mew=2)
    if title:
        ax.set_title(title)
    ax.set_xlabel("qx (px)")
    ax.set_ylabel("qy (px)")
    return ax


def plot_profile(x, y, ax=None, xlabel="q (1/Å)", ylabel="I", label=None,
                 title=None, **kw):
    """Plot a 1D profile (I(q), φ(q), G(r), ...)."""
    ax = _ax(ax)
    ax.plot(x, y, label=label, **kw)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if label:
        ax.legend()
    return ax


def plot_rdf(result, ax=None, title=None):
    """Plot G(r) from an :class:`~fourdstem.analysis.rdf.RDFResult`."""
    ax = _ax(ax)
    ax.plot(result.r, result.Gr, color="tab:blue")
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_xlabel("r (Å)")
    ax.set_ylabel("G(r)")
    ax.set_title(title or "Reduced density function")
    return ax


def plot_series_waterfall(profiles, coords, ax=None, offset=None,
                          xlabel="r (Å)", cmap="viridis"):
    """Stacked (waterfall) plot of per-frame profiles colored by coordinate."""
    import matplotlib.pyplot as plt

    ax = _ax(ax)
    coords = np.asarray(coords, float)
    finite = coords[np.isfinite(coords)]
    norm = plt.Normalize(finite.min(), finite.max()) if finite.size else None
    cm = plt.get_cmap(cmap)
    if offset is None:
        amp = np.nanmax([np.nanmax(np.abs(y)) for _, y in profiles]) or 1.0
        offset = 0.5 * amp
    for i, ((x, y), c) in enumerate(zip(profiles, coords)):
        color = cm(norm(c)) if (norm and np.isfinite(c)) else None
        ax.plot(x, np.asarray(y) + i * offset, color=color, lw=1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("stacked (offset by frame)")
    return ax


def plot_components(result, ncols=4, log=True, cmap="magma"):
    """Grid of decomposition component patterns with their loading maps.

    Returns the matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    comps = result.components
    loads = result.loadings
    k = comps.shape[0]
    two_d_loadings = loads.ndim == 3
    rows = 2 if two_d_loadings else 1
    fig, axes = plt.subplots(rows, k, figsize=(3 * k, 3 * rows), squeeze=False)
    for i in range(k):
        c = comps[i]
        disp = np.log1p(np.clip(c - np.nanmin(c), 0, None)) if log else c
        axes[0][i].imshow(disp, cmap=cmap)
        axes[0][i].set_title(f"comp {i}")
        axes[0][i].axis("off")
        if two_d_loadings:
            axes[1][i].imshow(loads[i], cmap="viridis")
            axes[1][i].set_title(f"loading {i}")
            axes[1][i].axis("off")
    fig.tight_layout()
    return fig


def plot_track(track, ax=None, ylabel="peak position (Å)", title=None):
    """Plot a tracked quantity vs. series coordinate (from insitu.tracking)."""
    ax = _ax(ax)
    ax.plot(track["coord"], track.get("position", track.get("integral")),
            "o-", color="tab:red")
    ax.set_xlabel("coordinate (T / t)")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    return ax
