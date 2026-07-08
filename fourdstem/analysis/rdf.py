"""
fourdstem.analysis.rdf
=====================
Reduced density function G(r) from a diffraction pattern — the amorphous /
electron-PDF pipeline, refactored from the original ``batch_dm4_to_rdf`` script
into composable pieces:

    scattering_terms      <f²>, <f>² for a composition (Kirkland factors)
    reduce_intensity      I(q) -> reduced structure factor φ(q), auto-fit scale N
    sine_ft               φ(q) -> G(r) via damped sine transform
    pattern_to_rdf        one-call: pattern (+center/mask) -> RDFResult

Conventions (verify against your own reference):
    * q = 1/d (crystallographic), Å⁻¹
    * G(r) = 8π ∫ q·φ(q)·sin(2π q r) dq
    * Lorch or Gaussian damping to suppress termination ripples

Scattering factors come from abTEM's ``kirkland.json`` if importable; otherwise
a crude analytic fallback runs (peak *positions* stay meaningful, absolute
amplitudes do not — a loud warning is emitted once).
"""
from __future__ import annotations
import os
import json
import warnings
from dataclasses import dataclass, field
import numpy as np

from ..utils.helpers import trapezoid
from ..analysis.azimuthal import azimuthal_integrate
from ..preprocess.center import find_center
from ..preprocess.masks import beam_stopper_mask


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass
class RDFConfig:
    """Reduction parameters. Lock these across an in-situ series so G(r)'s stay
    comparable; only beam center and scale N should vary per frame."""
    composition: dict = field(default_factory=lambda: {"Si": 1, "O": 2})
    q_int_min: float = 0.8       # FT lower limit (Å⁻¹)
    q_int_max: float = 12.0      # FT upper limit (Å⁻¹)
    r_max: float = 10.0          # G(r) grid max (Å)
    dr: float = 0.02             # G(r) step (Å)
    r_min: float = 1.10          # straight-line region below first shell (Å)
    damping: str = "lorch"       # "lorch" | "gauss" | "none"
    damping_b: float = 0.0       # for gauss: exp(-b q²)


@dataclass
class RDFResult:
    q: np.ndarray
    Iq: np.ndarray
    q_reduced: np.ndarray
    phi: np.ndarray
    r: np.ndarray
    Gr: np.ndarray
    N: float
    center: tuple | None = None
    diagnostics: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# scattering factors (Kirkland via abTEM; analytic fallback)
# ---------------------------------------------------------------------------
_Z = {"H": 1, "C": 6, "N": 7, "O": 8, "Si": 14, "Al": 13, "Au": 79,
      "Ti": 22, "Fe": 26, "Cu": 29, "Ge": 32}
_warned_sf = [False]


def _kirkland_params():
    import abtem
    base = os.path.dirname(abtem.__file__)
    for root, _, files in os.walk(base):
        if "kirkland.json" in files:
            with open(os.path.join(root, "kirkland.json")) as fh:
                return json.load(fh)
    raise FileNotFoundError("kirkland.json not found in abtem")


def _f_kirkland(q, p):
    # 12-coefficient Kirkland form; verify the coefficient order in your json.
    a1, b1, a2, b2, a3, b3, c1, d1, c2, d2, c3, d3 = p
    q2 = q * q
    return (a1 / (q2 + b1) + a2 / (q2 + b2) + a3 / (q2 + b3)
            + c1 * np.exp(-d1 * q2) + c2 * np.exp(-d2 * q2) + c3 * np.exp(-d3 * q2))


def scattering_terms(q, composition):
    """Return ``(<f²>, <f>²)`` for a composition, weighted by atomic fraction."""
    try:
        tbl = _kirkland_params()

        def fe(sym):
            key = sym if sym in tbl else str(_Z.get(sym, sym))
            return _f_kirkland(q, tbl[key])
    except Exception:
        if not _warned_sf[0]:
            warnings.warn(
                "abTEM/kirkland.json unavailable — using CRUDE analytic f(q). "
                "Peak positions OK; amplitudes NOT quantitative. Install abtem "
                "or supply real scattering factors for production."
            )
            _warned_sf[0] = True

        def fe(sym):
            Z = _Z.get(sym, 8)
            return Z * np.exp(-1.5 * q * q) + 0.02 * Z

    syms = list(composition)
    c = np.array([composition[s] for s in syms], float)
    c = c / c.sum()
    F = np.stack([fe(s) for s in syms], 0)          # (n_elem, n_q)
    f_avg_sq = (c[:, None] * F).sum(0) ** 2
    f_sq = (c[:, None] * F * F).sum(0)
    return f_sq, f_avg_sq


# ---------------------------------------------------------------------------
# damping + sine transform
# ---------------------------------------------------------------------------
def damping_window(q, cfg: RDFConfig):
    """Termination-ripple damping window over q."""
    if cfg.damping == "lorch":
        x = np.pi * q / cfg.q_int_max
        w = np.ones_like(q)
        nz = x > 1e-6
        w[nz] = np.sin(x[nz]) / x[nz]
        return w
    if cfg.damping == "gauss":
        return np.exp(-cfg.damping_b * q * q)
    return np.ones_like(q)


def sine_ft(q, phi, r, cfg: RDFConfig):
    """Damped sine Fourier transform: G(r) = 8π ∫ q φ(q) w(q) sin(2π q r) dq."""
    w = damping_window(q, cfg)
    integrand = q * phi * w
    return 8 * np.pi * np.array(
        [trapezoid(integrand * np.sin(2 * np.pi * q * ri), q) for ri in r]
    )


# ---------------------------------------------------------------------------
# reduction (auto-fit N)
# ---------------------------------------------------------------------------
def reduce_intensity(q, Iq, cfg: RDFConfig):
    """I(q) -> reduced structure factor φ(q), auto-fitting the scale N.

    N (∝ thickness × dose) is chosen to make G(r) straightest below ``r_min``
    (no atoms there). Returns ``(q_reduced, phi, r, Gr, diagnostics)``.
    """
    from scipy.optimize import minimize_scalar

    m = np.isfinite(Iq) & (q >= cfg.q_int_min) & (q <= cfg.q_int_max)
    qf, If = q[m], Iq[m]
    f_sq, f_avg_sq = scattering_terms(qf, cfg.composition)
    r = np.arange(0.0, cfg.r_max, cfg.dr)
    sub = r < cfg.r_min

    def phi_of_N(N):
        return (If - N * f_sq) / (N * f_avg_sq + 1e-30)

    def cost(logN):
        N = np.exp(logN)
        Gr = sine_ft(qf, phi_of_N(N), r[sub], cfg)
        A = np.vstack([r[sub], np.ones_like(r[sub])]).T
        res = Gr - A @ np.linalg.lstsq(A, Gr, rcond=None)[0]
        return np.sqrt(np.mean(res ** 2))

    tail = max(5, len(If) // 10)
    N0 = np.nanmedian(If[-tail:]) / max(np.nanmedian(f_sq[-tail:]), 1e-9)
    sol = minimize_scalar(cost, bracket=(np.log(N0 * 0.3), np.log(max(N0, 1e-6))),
                          method="brent", options=dict(xtol=1e-4))
    N = float(np.exp(sol.x))
    phi = phi_of_N(N)
    Gr = sine_ft(qf, phi, r, cfg)
    diag = {"N": N, "sub_rmin_rms": float(cost(np.log(N)))}
    return qf, phi, r, Gr, diag


# ---------------------------------------------------------------------------
# one-call pipeline
# ---------------------------------------------------------------------------
def pattern_to_rdf(pattern, q_per_px, cfg=None, center=None, mask=None,
                   stopper_coords=None):
    """Full pattern -> G(r) in one call.

    Parameters
    ----------
    pattern : 2D array
        Mean diffraction pattern.
    q_per_px : float
        Å⁻¹ per pixel.
    cfg : RDFConfig, optional
    center : (cx, cy), optional
        If None, fit via Friedel symmetry.
    mask : 2D bool array, optional
        Extra exclusion mask (e.g. Bragg spots). OR-combined with the stopper.
    stopper_coords : (x0,y0,x1,y1), optional
        Fixed beam-stopper box; if None the stopper is auto-detected.

    Returns
    -------
    RDFResult
    """
    cfg = cfg or RDFConfig()
    pattern = np.asarray(pattern, float)

    stop = beam_stopper_mask(pattern, stopper_coords)
    full_mask = stop if mask is None else (stop | np.asarray(mask, bool))

    if center is None:
        center, fried = find_center(pattern, full_mask)
    else:
        fried = float("nan")

    H, W = pattern.shape
    cx, cy = center
    q_edge = min(cx, cy, W - cx, H - cy) * q_per_px
    q_top = min(cfg.q_int_max * 1.05, q_edge)
    q_grid = np.arange(0.0, q_top, q_per_px)
    q, Iq = azimuthal_integrate(pattern, center, q_per_px, full_mask, q_grid)
    qf, phi, r, Gr, diag = reduce_intensity(q, Iq, cfg)
    diag["friedel_corr"] = fried
    return RDFResult(q=q, Iq=Iq, q_reduced=qf, phi=phi, r=r, Gr=Gr,
                     N=diag["N"], center=(float(cx), float(cy)), diagnostics=diag)
