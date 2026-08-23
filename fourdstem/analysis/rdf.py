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
_SF_CACHE = [None]
_BUNDLED_KIRKLAND = os.path.join(os.path.dirname(__file__), "data", "kirkland.json")


def _kirkland_params():
    """Load the Kirkland parameter table.

    Prefers the table bundled with fourdstem (so no external dependency is
    needed); falls back to abTEM's copy if a symbol is missing. Both use the
    ``[[a1,a2,a3],[b1,b2,b3],[c1,c2,c3],[d1,d2,d3]]`` layout.
    """
    if _SF_CACHE[0] is not None:
        return _SF_CACHE[0]
    tbl = {}
    try:
        with open(_BUNDLED_KIRKLAND) as fh:
            tbl = {k: v for k, v in json.load(fh).items() if not k.startswith("_")}
    except Exception:
        tbl = {}
    try:                                    # merge abTEM's table if available
        import abtem
        base = os.path.dirname(abtem.__file__)
        for root, _, files in os.walk(base):
            if "kirkland.json" in files:
                with open(os.path.join(root, "kirkland.json")) as fh:
                    for k, v in json.load(fh).items():
                        tbl.setdefault(k, v)
                break
    except Exception:
        pass
    if not tbl:
        raise FileNotFoundError("no Kirkland parameter table available")
    _SF_CACHE[0] = tbl
    return tbl


def _f_kirkland(q, p):
    """Kirkland electron scattering factor from ``p = [[a],[b],[c],[d]]`` (len-3 each).

    f_e(q) = Σ a_i/(q²+b_i) + Σ c_i·exp(-d_i·q²),  q = 1/d in 1/Å.
    """
    a, b, c, d = p
    q2 = q * q
    f = np.zeros_like(q, dtype=float)
    for i in range(3):
        f = f + a[i] / (q2 + b[i])
    for i in range(3):
        f = f + c[i] * np.exp(-d[i] * q2)
    return f


def scattering_terms(q, composition):
    """Return ``(<f²>, <f>²)`` for a composition, weighted by atomic fraction."""
    try:
        tbl = _kirkland_params()
        missing = [s for s in composition if s not in tbl]
        if missing:
            raise KeyError(f"no Kirkland params for {missing}")

        def fe(sym):
            return _f_kirkland(q, tbl[sym])
    except Exception as e:
        if not _warned_sf[0]:
            warnings.warn(
                f"Kirkland scattering factors unavailable ({e}) — using a CRUDE "
                "analytic f(q). Peak positions OK; amplitudes NOT quantitative."
            )
            _warned_sf[0] = True

        def fe(sym):                        # gentle Mott-like decay (approximate)
            Z = _Z.get(sym, 8)
            return Z / (1.0 + (q / 0.3) ** 2) + 0.1 * Z * np.exp(-2.0 * q * q)

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
    if m.sum() < 5:
        q_lo = float(np.nanmin(q)) if q.size else float("nan")
        q_hi = float(np.nanmax(q)) if q.size else float("nan")
        raise ValueError(
            "reduction window selects too few points "
            f"({int(m.sum())}). The FT window is "
            f"q_int_min={cfg.q_int_min}, q_int_max={cfg.q_int_max} (1/A) but the "
            f"data only spans q = {q_lo:.4g} .. {q_hi:.4g} 1/A. This almost always "
            "means the q-calibration (q_per_px) is wrong or in the wrong unit. "
            "Check q_per_px against a known ring (see calibration helpers), pass a "
            "q_unit_hint when loading, or set q_int_min/q_int_max to match your q "
            "range."
        )
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
    if not np.isfinite(N0) or N0 <= 0:            # keep the log-bracket valid
        N0 = 1.0
    sol = minimize_scalar(cost, bracket=(np.log(N0 * 0.3), np.log(max(N0, 1e-6))),
                          method="brent", options=dict(xtol=1e-4))
    N = float(np.exp(sol.x))
    phi = phi_of_N(N)
    Gr = sine_ft(qf, phi, r, cfg)
    diag = {"N": N, "sub_rmin_rms": float(cost(np.log(N)))}
    return qf, phi, r, Gr, diag


def _reduce_row(payload):
    """Picklable worker: reduce one I(q) profile to (phi, Gr, N). None on failure."""
    Iq, q, cfg = payload
    try:
        qf, phi, r, Gr, diag = reduce_intensity(q, Iq, cfg)
        return phi, Gr, float(diag["N"])
    except Exception:
        return None


def reduce_profiles(profiles, q, cfg: RDFConfig, n_jobs=1, progress=False):
    """Reduce a *stack* of per-position I(q) profiles to φ(q) and G(r).

    For a whole NBED scan you get one radial profile per probe position
    (:func:`radial_profiles`); this reduces every one to a background-removed
    structure factor φ(q) and its sine-FT G(r), so you can then decompose the
    stack (NMF/PCA) into a few characteristic structure factors / RDFs.

    Parameters
    ----------
    profiles : (n_pos, n_q) array
        Per-position I(q) (finite; e.g. from ``radial_profiles``).
    q : (n_q,) array
        Common q axis (1/Å).
    cfg : RDFConfig
    n_jobs : int
        Parallel workers (``-1`` = all cores). Each row runs an independent
        scale-fit reduction, so this fans out cleanly.
    progress : bool

    Returns
    -------
    dict with ``q`` (windowed q), ``phi`` (n_pos, nq), ``r``, ``Gr`` (n_pos, nr),
    ``N`` (n_pos,), ``ok`` (n_pos bool — False where the reduction failed).
    """
    from ..utils.parallel import parallel_map

    profiles = np.asarray(profiles, float)
    q = np.asarray(q, float)
    n = profiles.shape[0]
    m = np.isfinite(q) & (q >= cfg.q_int_min) & (q <= cfg.q_int_max)
    qf = q[m]
    r = np.arange(0.0, cfg.r_max, cfg.dr)

    out = parallel_map(_reduce_row, [(profiles[i], q, cfg) for i in range(n)],
                       n_jobs=n_jobs, progress=progress, desc="reduce")

    phi = np.full((n, qf.size), np.nan)
    Gr = np.full((n, r.size), np.nan)
    N = np.full(n, np.nan)
    ok = np.zeros(n, bool)
    for i, res in enumerate(out):
        if res is None:
            continue
        p, g, nn = res
        if p.shape[0] == qf.size and g.shape[0] == r.size:
            phi[i], Gr[i], N[i], ok[i] = p, g, nn, True
    return dict(q=qf, phi=phi, r=r, Gr=Gr, N=N, ok=ok)


# ---------------------------------------------------------------------------
# one-call pipeline
# ---------------------------------------------------------------------------
def pattern_to_rdf(pattern, q_per_px, cfg=None, center=None, mask=None,
                   stopper_coords=None, center_beam_radius=None):
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
    center_beam_radius : float, optional
        If given, exclude a central disk of this radius (px) around the beam
        center — removes the intense direct/transmitted beam so it can't bias the
        low-q intensity. A good starting value is ``q_int_min / q_per_px``.

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

    if center_beam_radius:
        from ..preprocess.masks import disk_mask
        full_mask = full_mask | disk_mask(pattern.shape, center, center_beam_radius)

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


def save_rdf(path, result: RDFResult, **extra):
    """Save an :class:`RDFResult` to a compressed .npz (with optional extras).

    Stores the full pipeline (q, Iq, q_reduced, phi, r, Gr) plus scalars, so a
    later notebook can reload the whole temperature series for NMF. Pass e.g.
    ``temperature=450`` or ``source="scan.dm4"`` as extra fields.
    """
    from ..io.writers import save_result_npz

    payload = dict(
        q=result.q, Iq=result.Iq, q_reduced=result.q_reduced, phi=result.phi,
        r=result.r, Gr=result.Gr, N=result.N,
        center=np.array(result.center if result.center is not None
                        else [np.nan, np.nan]),
        friedel_corr=result.diagnostics.get("friedel_corr", np.nan),
        sub_rmin_rms=result.diagnostics.get("sub_rmin_rms", np.nan),
    )
    payload.update(extra)
    return save_result_npz(path, **payload)


def load_rdf(path):
    """Load an .npz written by :func:`save_rdf` into a plain dict of arrays."""
    from ..io.writers import load_result_npz
    return load_result_npz(path)


def rdf_quality(result, expected_first_peak=1.61, first_win=(1.45, 1.85)):
    """Objective reduction-quality metrics for an :class:`RDFResult`.

    Returns a dict of numbers + boolean ``flags`` and a human ``verdict``. There
    is no curve "fit" — the reduction fits one scale N to flatten the low-r
    region — so quality means: is the low-r region flat, does φ(q) oscillate
    around 0 without ramping, and does the first peak land where expected?

    Metrics
    -------
    first_peak_r / first_peak_offset : G(r) first-shell position and its offset
        from ``expected_first_peak`` (Å).
    low_r_rms : straight-line residual of G(r) below r_min (self-consistency;
        lower is better).
    phi_lowr_slope_ok : whether G(r) below the first bond has no positive bump.
    phi_highq_abs : mean |φ| in the top q-decile (should be small; a large value
        means φ ramps → bad scattering factors / calibration).
    qmax : maximum q used (Å⁻¹, q=1/d) — sets the real-space resolution.
    """
    from .peaks import first_peak_position

    r = np.asarray(result.r); Gr = np.asarray(result.Gr)
    q = np.asarray(result.q_reduced); phi = np.asarray(result.phi)
    r1, _ = first_peak_position(r, Gr, *first_win)
    low_r_rms = float(result.diagnostics.get("sub_rmin_rms", np.nan))
    # a "ramp" is a signed high-q DRIFT away from 0 (φ should oscillate about 0),
    # measured relative to the oscillation amplitude — not just a large |φ|.
    hi = q >= np.percentile(q, 85)
    phi_std = float(np.nanstd(phi)) + 1e-9
    phi_drift = float(np.nanmean(phi[hi]) / phi_std) if hi.any() else float("nan")
    qmax = float(np.nanmax(q)) if q.size else float("nan")
    # low-r spurious positive bump (below ~1.0 Å there should be none)
    sub = r < min(1.0, first_win[0] - 0.3)
    bump = float(np.nanmax(Gr[sub])) if sub.any() else 0.0
    peak_h = float(np.nanmax(Gr[(r >= first_win[0]) & (r <= first_win[1])]) or 1.0)
    flags = {
        "first_peak_ok": abs(r1 - expected_first_peak) < 0.15,
        "phi_not_ramping": abs(phi_drift) < 1.5,
        "low_r_clean": bump < 0.25 * abs(peak_h),
    }
    verdict = "good" if all(flags.values()) else "check"
    return dict(first_peak_r=float(r1),
                first_peak_offset=float(r1 - expected_first_peak),
                low_r_rms=low_r_rms, phi_highq_drift=phi_drift, qmax=qmax,
                low_r_bump_frac=float(bump / abs(peak_h)), N=float(result.N),
                flags=flags, verdict=verdict)
