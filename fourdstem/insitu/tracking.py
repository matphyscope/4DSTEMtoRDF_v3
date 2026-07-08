"""
fourdstem.insitu.tracking
=========================
Track a feature (a peak position, a peak height, a coordination-like integral)
across the frames of an in-situ :class:`~fourdstem.insitu.series.Series`.

The pattern is: compute a 1D profile per frame (I(q), reduced φ(q), or G(r)),
then follow a chosen peak's position/amplitude versus the series coordinate —
e.g. first-neighbour distance r₁ vs. temperature.
"""
from __future__ import annotations
import numpy as np

from ..analysis.peaks import first_peak_position, peak_centroid, find_peaks_1d


def track_peak(profiles, coords, x_window, mode="max", refine=True):
    """Follow one peak across a set of 1D profiles.

    Parameters
    ----------
    profiles : list of (x, y)
        One ``(x_axis, y_values)`` tuple per frame (same-length x not required).
    coords : array
        Series coordinate for each profile (T, t, ...).
    x_window : (x_min, x_max)
        Search window in which the peak lives.
    mode : {"max", "centroid"}
        ``max`` tracks the tallest local maximum (parabolically refined);
        ``centroid`` tracks the intensity-weighted centroid (robust for broad,
        noisy shells).

    Returns
    -------
    dict with arrays ``coord``, ``position``, ``amplitude``.
    """
    x_min, x_max = x_window
    pos, amp = [], []
    for (x, y) in profiles:
        if mode == "centroid":
            xp = peak_centroid(x, y, x_min, x_max)
            # amplitude at the centroid (nearest sample)
            if np.isnan(xp):
                yp = np.nan
            else:
                yp = float(np.interp(xp, x, y))
        else:
            xp, yp = first_peak_position(x, y, x_min, x_max, refine=refine)
        pos.append(xp)
        amp.append(yp)
    return {
        "coord": np.asarray(coords, float),
        "position": np.asarray(pos, float),
        "amplitude": np.asarray(amp, float),
    }


def track_multiple_peaks(profiles, coords, windows, mode="max"):
    """Track several peaks at once. ``windows`` is a dict ``{name: (xmin, xmax)}``.

    Returns ``{name: track_dict}``.
    """
    return {name: track_peak(profiles, coords, win, mode=mode)
            for name, win in windows.items()}


def integrate_region(profiles, coords, x_window, baseline="none"):
    """Integrate each profile over ``x_window`` vs. the series coordinate.

    A proxy for shell area / coordination trend. ``baseline="linear"`` subtracts
    a straight line between the window endpoints before integrating.

    Returns ``dict(coord, integral)``.
    """
    from ..utils.helpers import trapezoid

    x_min, x_max = x_window
    out = []
    for (x, y) in profiles:
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        sel = (x >= x_min) & (x <= x_max)
        if sel.sum() < 2:
            out.append(np.nan)
            continue
        xs, ys = x[sel], y[sel]
        if baseline == "linear":
            m = (ys[-1] - ys[0]) / (xs[-1] - xs[0] + 1e-30)
            base = ys[0] + m * (xs - xs[0])
            ys = ys - base
        out.append(float(trapezoid(ys, xs)))
    return {"coord": np.asarray(coords, float), "integral": np.asarray(out, float)}
