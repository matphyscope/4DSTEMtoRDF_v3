"""
fourdstem.analysis.peaks
=======================
Peak finding and characterization on 1D profiles (I(q), G(r), reduced φ(q)).

For 2D Bragg-spot detection see ``fourdstem.preprocess.masks.detect_bragg_peaks``.
This module answers "where are the peaks in this radial profile, and how do they
move" — the workhorse for tracking first/second-neighbour distances vs.
temperature or time in an in-situ series.
"""
from __future__ import annotations
import numpy as np


def find_peaks_1d(x, y, height=None, prominence=None, distance=None,
                  smooth=0):
    """Locate peaks in a 1D profile ``y(x)``.

    Thin wrapper over :func:`scipy.signal.find_peaks` that returns results in the
    ``x`` coordinate (not just indices). ``smooth`` applies a moving average of
    that odd window before detection.

    Returns a list of ``dict(x, y, index, prominence)`` sorted by descending y.
    """
    from scipy.signal import find_peaks

    x = np.asarray(x, float)
    yv = np.asarray(y, float)
    if smooth and smooth > 1:
        k = np.ones(smooth) / smooth
        yv = np.convolve(yv, k, mode="same")

    kw = {}
    if height is not None:
        kw["height"] = height
    if prominence is not None:
        kw["prominence"] = prominence
    if distance is not None:
        kw["distance"] = distance
    idx, props = find_peaks(yv, **kw)

    proms = props.get("prominences", np.full(idx.shape, np.nan))
    out = [dict(x=float(x[i]), y=float(yv[i]), index=int(i),
                prominence=float(p)) for i, p in zip(idx, proms)]
    out.sort(key=lambda p: -p["y"])
    return out


def refine_peak_parabolic(x, y, index):
    """Sub-sample peak position/height by fitting a parabola to 3 points.

    Returns ``(x_peak, y_peak)``. Falls back to the sample point at the edges.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    i = int(index)
    if i <= 0 or i >= len(y) - 1:
        return float(x[i]), float(y[i])
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    denom = (y0 - 2 * y1 + y2)
    if abs(denom) < 1e-30:
        return float(x[i]), float(y[i])
    delta = 0.5 * (y0 - y2) / denom          # in index units, [-0.5, 0.5]
    dx = x[i + 1] - x[i]
    x_peak = x[i] + delta * dx
    y_peak = y1 - 0.25 * (y0 - y2) * delta
    return float(x_peak), float(y_peak)


def first_peak_position(x, y, x_min=None, x_max=None, refine=True):
    """Position of the tallest peak within ``[x_min, x_max]`` (e.g. first RDF shell).

    Returns ``(x_peak, y_peak)`` or ``(nan, nan)`` if nothing qualifies.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    sel = np.ones(x.shape, bool)
    if x_min is not None:
        sel &= x >= x_min
    if x_max is not None:
        sel &= x <= x_max
    if not sel.any():
        return float("nan"), float("nan")
    idx_local = np.argmax(np.where(sel, y, -np.inf))
    if refine:
        return refine_peak_parabolic(x, y, idx_local)
    return float(x[idx_local]), float(y[idx_local])


def fit_gaussian_peak(x, y, x_min, x_max, with_offset=True):
    """Least-squares Gaussian fit of a peak within ``[x_min, x_max]``.

    Models ``y ≈ A·exp(-(x-mu)²/(2σ²)) [+ c]``. This is the robust way to track a
    peak *position* (``mu``) across an in-situ series — e.g. the first RDF shell
    (Si–O ~1.6 Å) as a function of temperature — because it uses the whole peak
    shape, not just the argmax sample.

    Returns a dict ``{center, amplitude, sigma, offset, fwhm, success, xfit, yfit}``.
    ``center`` is ``nan`` if the window is empty or the fit fails.
    """
    from scipy.optimize import curve_fit

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    sel = (x >= x_min) & (x <= x_max) & np.isfinite(y)
    xs, ys = x[sel], y[sel]
    fail = dict(center=float("nan"), amplitude=float("nan"),
                sigma=float("nan"), offset=float("nan"), fwhm=float("nan"),
                success=False, xfit=xs, yfit=np.full_like(xs, np.nan))
    if xs.size < (4 if with_offset else 3):
        return fail

    c0 = float(np.min(ys)) if with_offset else 0.0
    A0 = float(np.max(ys) - c0)
    mu0 = float(xs[np.argmax(ys)])
    sig0 = max((x_max - x_min) / 6.0, (xs[1] - xs[0]) if xs.size > 1 else 0.1)

    if with_offset:
        def model(xx, A, mu, sig, c):
            return A * np.exp(-((xx - mu) ** 2) / (2 * sig ** 2)) + c
        p0 = [A0, mu0, sig0, c0]
    else:
        def model(xx, A, mu, sig):
            return A * np.exp(-((xx - mu) ** 2) / (2 * sig ** 2))
        p0 = [A0, mu0, sig0]

    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")          # benign OptimizeWarning on flat peaks
            popt, _ = curve_fit(model, xs, ys, p0=p0, maxfev=10000)
    except Exception:
        return fail
    A, mu, sig = popt[0], popt[1], abs(popt[2])
    c = popt[3] if with_offset else 0.0
    return dict(center=float(mu), amplitude=float(A), sigma=float(sig),
                offset=float(c), fwhm=float(2.3548 * sig), success=True,
                xfit=xs, yfit=model(xs, *popt))


def peak_centroid(x, y, x_min, x_max):
    """Intensity-weighted centroid of ``y`` over ``[x_min, x_max]``.

    More robust than the argmax for broad, noisy shells. Returns ``nan`` if the
    window has no positive weight.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    sel = (x >= x_min) & (x <= x_max)
    w = np.clip(y[sel], 0, None)
    if w.sum() <= 0:
        return float("nan")
    return float((x[sel] * w).sum() / w.sum())
