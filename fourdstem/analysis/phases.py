"""
fourdstem.analysis.phases
==========================
High-level, general-purpose SEI phase screening from NBD signals — the clean
"engine" behind the phase-mapping notebook. Given the two summary patterns that
every 4D-STEM scan produces — the **mean** (amorphous halo) and the **max**
(polycrystalline rings + Bragg spots) — it answers, for each candidate phase:

    is it there? (verdict)   how strong? (score / spot count)   which rings?

and flags rings that no candidate explains. Naming overlapping Li phases is
fundamentally limited (see :func:`score_phases`), so every verdict carries an
honest confidence: a phase is only ``"confirmed"`` when it owns a ring/spot that
no other candidate can produce.

The spatial ("where") and cepstral ("how many distinct regions") parts need the
full cube and live in :func:`analyze_phases`; the diffraction-only screen here
runs on the 2-D patterns alone, so it works on an exported mean/max image too.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from .unmix import COMPOUND_RINGS
from .peaks import find_fsdp

CANDIDATES = ["LiF", "Li2O", "Li3N", "Li2CO3", "Li2S"]

# Diagnostic real-space interatomic distance per phase (A), for a per-phase
# cepstral fluctuation map. LiF/Li2O/Li3N share ~2.0 A (nearest neighbour) so
# their cepstral maps look alike — the honest resolution overlap; Li2CO3 (short
# C-O 1.28) and Li2S (Li-S 2.47) sit at distinct distances and do separate.
PHASE_DISTANCE = {"LiF": 2.01, "Li2O": 2.00, "Li3N": 1.94,
                  "Li2CO3": 1.28, "Li2S": 2.47}


# ---------------------------------------------------------------- data classes
@dataclass
class PhaseEvidence:
    phase: str
    verdict: str                       # 'confirmed' | 'possible' | 'weak/absent'
    score: float                       # 0..1 weighted ring-match fraction
    matched_d: list = field(default_factory=list)     # measured d that fit (A)
    unique_d: list = field(default_factory=list)       # d only this phase explains
    missing_strong_d: list = field(default_factory=list)
    n_spots: int = 0                   # detected spots on this phase's rings
    n_unique_spots: int = 0            # spots at a d only this phase can own
    amount: float = float("nan")       # mean thickness-normalized DF over material
    diag_d: float = float("nan")       # d (A) of the ring used for the location map


@dataclass
class DiffractionReport:
    center: tuple
    q_per_px: float
    halo_q: float                      # amorphous FSDP position (1/A), nan if none
    halo_conf: float
    rings_d: list                      # measured ring d-spacings (A), from max radial
    n_spots: int                       # total detected Bragg spots
    crystallinity: float               # spot signal / background (unitless)
    phases: dict                       # name -> PhaseEvidence
    unexplained_d: list                # measured rings no candidate explains

    def summary(self):
        print(f"center=({self.center[0]:.0f},{self.center[1]:.0f}) | "
              f"q_per_px={self.q_per_px:.5g} 1/A/px")
        hc = f"conf {self.halo_conf:.1f}" if np.isfinite(self.halo_q) else "none"
        print(f"amorphous FSDP: q={self.halo_q:.3f} (d={1/self.halo_q:.2f} A) [{hc}]"
              if np.isfinite(self.halo_q) else "amorphous FSDP: none (flat halo)")
        print(f"crystallinity: {self.crystallinity:.1f}  | "
              f"{self.n_spots} spots | rings d(A)={[round(d,2) for d in self.rings_d]}")
        print("phase verdicts (confirmed = owns a ring no other candidate explains):")
        for c in self.phases.values():
            print(f"  {c.phase:7s} {c.verdict:14s} score {c.score:.2f} | "
                  f"unique-ring d={c.unique_d} | matched {[round(x,2) for x in c.matched_d]} | "
                  f"missing-strong {c.missing_strong_d} | spots {c.n_spots} ({c.n_unique_spots} unique)")
        if self.unexplained_d:
            print(f"unexplained rings d(A)={[round(d,2) for d in self.unexplained_d]} "
                  f"-> phase outside the candidate set?")


# ----------------------------------------------------------------- primitives
def _resolve_center(pat, center):
    if center is not None:
        return float(center[0]), float(center[1])
    thr = np.percentile(pat, 99.5)
    ys, xs = np.where(pat >= thr)
    if xs.size < 5:
        h, w = pat.shape
        return w / 2.0, h / 2.0
    return float(xs.mean()), float(ys.mean())


def _radial_median(pat, center, n_bin, r_lo, r_hi):
    h, w = pat.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(xx - center[0], yy - center[1])
    edges = np.linspace(0, r_hi, n_bin + 1)
    idx = np.digitize(r.ravel(), edges) - 1
    g = pat.ravel()
    prof = np.full(n_bin, np.nan)
    for b in range(n_bin):
        sel = g[idx == b]
        if sel.size:
            prof[b] = np.median(sel)
    rc = 0.5 * (edges[:-1] + edges[1:])
    return rc, prof


def detect_rings(max_pat, center, q_per_px, q_beam=0.20, q_max=1.0,
                 n_bin=200, top_n=8, nsig=1.5):
    """Ring d-spacings (A) from a max-projection radial profile.

    Robust: fits a broad background, keeps residual bumps ``nsig`` above the
    noise, inside a physical window ``[q_beam, q_max]`` (beam core and far-field
    noise excluded), and returns at most ``top_n`` strongest. Works whether there
    are several sharp rings or almost none (returns few/empty then).
    """
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import find_peaks
    rc, prof = _radial_median(
        max_pat, center, n_bin, 0,
        min(center[0], center[1], max_pat.shape[1] - center[0],
            max_pat.shape[0] - center[1]))
    q = rc * q_per_px
    prof = np.nan_to_num(prof, nan=np.nanmedian(prof))
    base = uniform_filter1d(prof, max(5, n_bin // 12))
    res = prof - base
    m = (q >= q_beam) & (q <= q_max)
    noise = np.median(np.abs(res[m] - np.median(res[m]))) * 1.4826 + 1e-9
    pk, props = find_peaks(np.where(m, res, -np.inf), prominence=nsig * noise, distance=3)
    order = np.argsort(props["prominences"])[::-1][:top_n]
    return [float(q[pk[i]]) for i in order]           # ring q (1/A), strongest first


def detect_spots(max_pat, center, q_per_px, q_beam=0.15, q_max=1.15,
                 n_mad=8.0, min_dist=3, tophat=11):
    """Bragg spots as ``[(x, y, q), ...]`` using a noise-relative threshold.

    A white top-hat removes the smooth beam halo; only local maxima above
    ``median + n_mad*MAD`` of the top-hat inside the working annulus survive, so
    the diffuse noise field is not mistaken for spots.
    """
    from scipy.ndimage import white_tophat, maximum_filter
    dp = max_pat.shape
    yy, xx = np.mgrid[0:dp[0], 0:dp[1]]
    rr = np.hypot(xx - center[0], yy - center[1]) * q_per_px
    th = white_tophat(max_pat, size=tophat)
    th[(rr < q_beam) | (rr > q_max)] = 0
    ann = (rr >= q_beam) & (rr <= q_max)
    vv = th[ann]
    med = np.median(vv)
    mad = 1.4826 * np.median(np.abs(vv - med)) + 1e-9
    loc = (th == maximum_filter(th, size=2 * min_dist + 1)) & (th > med + n_mad * mad)
    ys, xs = np.where(loc)
    return [(int(x), int(y), float(np.hypot(x - center[0], y - center[1]) * q_per_px))
            for x, y in zip(xs, ys)]


def ring_azimuthal_spots(pattern, center, q_per_px, ring_q, dq=0.03, n_theta=360,
                         max_spots=10, min_prom_frac=0.06, min_sep_deg=12.0,
                         min_nsig=4.0, min_contrast=0.0):
    """Spots on one diffraction ring, found by unwrapping the ring azimuthally.

    Polar-transforms ``pattern`` about ``center``, averages the intensity over the
    thin radial band ``ring_q ± dq`` (1/Å) to get an intensity-vs-angle profile
    (the ring "unrolled" into a 1-D line), and returns its prominent angular peaks
    as spots — up to ``max_spots`` per ring, separated by at least ``min_sep_deg``,
    each with prominence ≥ ``min_prom_frac`` of the profile's peak-to-trough range
    AND rising ``min_nsig`` robust MAD above the ring's *median* level. The
    MAD threshold adapts to each ring's own noise, so a low-contrast-but-clearly-
    discrete peak (a few % above the median but many σ above the noise) is kept,
    while a smooth, spot-free ring (only interpolation ripple / noise) yields none —
    unlike a fixed fractional-contrast cut, which misses real weak spots and can
    admit noise. ``min_contrast`` (default 0, off) optionally also requires the peak
    to exceed that fraction of the median. Wrap-around at 0/360° is handled.
    Returns ``(spots, theta_deg, profile)`` where
    ``spots`` is ``[(x, y, theta_rad, height), ...]`` and ``(theta_deg, profile)`` is
    the unrolled 1-D ring for plotting.
    """
    from scipy.signal import find_peaks
    from ..preprocess.transform import polar_transform

    center = _resolve_center(pattern, center)
    rq_px = float(ring_q) / q_per_px
    r_max = rq_px + max(3.0, 2.0 * dq / q_per_px)
    polar, r_ax, th_ax = polar_transform(pattern, center, n_theta=int(n_theta),
                                         r_max=r_max)
    band = np.abs(r_ax - rq_px) <= (dq / q_per_px)
    if not band.any():
        band = np.abs(r_ax - rq_px) <= (r_ax[1] - r_ax[0]) * 1.5
    prof = np.nanmean(np.where(np.isnan(polar[:, band]), np.nan, polar[:, band]), axis=1)
    prof = np.nan_to_num(prof, nan=float(np.nanmin(prof)) if np.isfinite(prof).any() else 0.0)
    theta_deg = np.degrees(th_ax)
    rng = float(prof.max() - prof.min())
    if rng <= 0:
        return [], theta_deg, prof
    # wrap-around: tile x3, detect in the middle copy
    n = prof.size
    med = float(np.median(prof))
    mad = 1.4826 * float(np.median(np.abs(prof - med))) + 1e-12
    thr = med + float(min_nsig) * mad                             # noise-adaptive floor
    if min_contrast > 0:
        thr = max(thr, med + float(min_contrast) * max(med, 1e-12))
    tiled = np.concatenate([prof, prof, prof])
    dist = max(1, int(round(min_sep_deg / 360.0 * n)))
    pk, props = find_peaks(tiled, prominence=min_prom_frac * rng, distance=dist)
    mid = [(int(p - n), float(props["prominences"][j]))
           for j, p in enumerate(pk)
           if n <= p < 2 * n and prof[int(p - n)] >= thr]          # significant peak
    mid.sort(key=lambda t: -prof[t[0]])
    spots = []
    for idx, _prom in mid[:max_spots]:
        th = th_ax[idx]
        x = center[0] + rq_px * np.cos(th)
        y = center[1] + rq_px * np.sin(th)
        spots.append((float(x), float(y), float(th), float(prof[idx])))
    return spots, theta_deg, prof


def score_phases(rings_q, spots_q, candidates=None, tol=0.045, min_unique_spots=5,
                 q_confirm_min=0.28, rings=None):
    """Per-phase verdict from measured ring/spot positions.

    Ownership is decided at the **measured** position, not the tabulated one: a
    measured ring/spot at ``q`` is *owned* by every candidate that has a ring
    within ``tol`` (the convergence-limited q resolution), and is **unique** when
    exactly one candidate owns it. A phase is ``confirmed`` when it uniquely owns
    a measured ring, or at least ``min_unique_spots`` Bragg spots sit at a
    d only it can produce; ``possible`` when it owns rings but none uniquely and
    no strong ring is missing; ``weak/absent`` otherwise. This uses the sparse
    spot signal (robust when the azimuthal ring is too weak to detect) and avoids
    the tabulated-position pitfall where a near-neighbour phase steals uniqueness.

    Only rings above ``q_confirm_min`` (small d) can *confirm* a phase: at low q
    the ``tol`` window spans a huge d-range, so a beam-tail/noise bump there
    "uniquely" matches whichever phase happens to own the sole large-d ring — a
    false positive. Such low-q rings still count toward ``matched``/``possible``.
    """
    tbl = rings if rings is not None else COMPOUND_RINGS
    names = list(candidates) if candidates is not None else list(CANDIDATES)
    meas = np.atleast_1d(np.asarray(rings_q, float))
    meas = meas[np.isfinite(meas)]
    spq = np.asarray([s if np.isscalar(s) else s[-1] for s in spots_q], float)

    def owners(qm):
        return [c for c in names if any(abs(1.0 / d - qm) <= tol for d, _ in tbl[c])]

    ring_owned = {c: [] for c in names}
    ring_unique = {c: [] for c in names}
    for qm in meas:
        ow = owners(qm)
        for c in ow:
            ring_owned[c].append(round(1.0 / qm, 2))
        if len(ow) == 1 and qm >= q_confirm_min:      # low-q uniqueness is unreliable
            ring_unique[ow[0]].append(round(1.0 / qm, 2))
    spot_owned = {c: 0 for c in names}
    spot_unique = {c: 0 for c in names}
    for qm in spq:
        ow = owners(qm)
        for c in ow:
            spot_owned[c] += 1
        if len(ow) == 1:
            spot_unique[ow[0]] += 1

    out = {}
    for c in names:
        crings = tbl[c]
        wsum = sum(w for _, w in crings) + 1e-12
        score = sum(w for d, w in crings
                    if meas.size and np.min(np.abs(meas - 1.0 / d)) <= tol) / wsum
        missing = [round(d, 2) for d, w in crings if w >= 0.6 and
                   (not meas.size or np.min(np.abs(meas - 1.0 / d)) > tol)]
        uniq = sorted(set(ring_unique[c]), reverse=True)
        nsp, nsp_u = spot_owned[c], spot_unique[c]
        if uniq or nsp_u >= min_unique_spots:
            verdict = "confirmed"
        elif ring_owned[c] and not missing:
            verdict = "possible"
        else:
            verdict = "weak/absent"
        out[c] = PhaseEvidence(c, verdict, round(score, 3),
                               sorted(set(ring_owned[c]), reverse=True), uniq,
                               missing, nsp, nsp_u)
    return out


def phase_ring_profile(q, compound, sigma_q=0.035):
    """Reference radial ring fingerprint ``R_p(q)`` for one compound.

    The tabulated ``(d, weight)`` powder lines of ``compound`` broadened by a
    Gaussian of width ``sigma_q`` (1/A) — matched to the convergence-limited q
    resolution (~2*alpha/lambda) plus intrinsic ring width. Returns an array like
    ``q``. Used as a basis vector for :func:`decompose_fractions`.
    """
    q = np.asarray(q, float)
    prof = np.zeros_like(q)
    for d, w in COMPOUND_RINGS[compound]:
        prof += w * np.exp(-0.5 * ((q - 1.0 / d) / sigma_q) ** 2)
    return prof


def _groups_by_correlation(labels, G, thr):
    """Union-find grouping of labels whose |Gram correlation| exceeds ``thr``."""
    n = len(labels)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if abs(G[i, j]) > thr:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(labels[i])
    return list(groups.values())


def _rolling_min_baseline(y, win):
    """Lower-envelope baseline: rolling minimum then smoothing.

    Follows a steep amorphous halo / beam tail (dipping under sharp rings) without
    leaving the positive low-q artifact a symmetric running-mean does on a steep
    monotonic decay. ``win`` in points.
    """
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    win = max(3, int(win) | 1)
    return uniform_filter1d(minimum_filter1d(y, win, mode="nearest"), win, mode="nearest")


def decompose_fractions(q, I, candidates=None, sigma_q=0.035, bg_win_frac=0.12,
                        q_lo=0.20, q_hi=None, group_corr=0.9,
                        halo_q=None, halo_sigma=0.08):
    """Linear (NNLS) phase-fraction decomposition of a radial diffraction profile.

    Thin-sample kinematic superposition: through a column where phases are stacked
    along the beam,

        ``I(q) ~ baseline(q) + h * H(q) + sum_p a_p * R_p(q)``,

    with ``R_p`` the crystalline ring fingerprint (:func:`phase_ring_profile`),
    ``a_p >= 0`` the column amount of phase p, and ``H(q)`` a broad **amorphous
    halo** basis (a wide Gaussian at the FSDP ``halo_q``). The halo term is the key
    to a real (mostly amorphous) SEI: without it NNLS mis-assigns the strong
    amorphous first-sharp-diffraction-peak to whichever candidate has a low-q ring
    nearby; with it, the phase fingerprints see only the sharp crystalline rings.
    A rolling-min lower-envelope baseline removes the smooth beam tail first.

    Fractions ``f_p = a_p / sum_p a_p`` give the **crystalline** composition from
    ring SHAPE, independent of the overall scale (thickness) — the "how much of
    each phase" a single-ring virtual image cannot give. ``crystallinity`` =
    crystalline peak energy / (crystalline + halo) says how much of the ordered
    signal is crystalline vs amorphous.

    **Degeneracy is reported, not hidden.** The normalized Gram matrix
    ``G_pp' = <R_p, R_p'>`` flags collinear fingerprints (the ~2 A LiF/Li2O/Li3N
    overlap); phases with ``|G| > group_corr`` are also merged into a **group**
    amount. Individual fractions are kept with that caveat.

    Knobs: ``sigma_q`` (ring width / resolution), ``bg_win_frac`` (baseline window
    fraction), ``q_lo``/``q_hi`` (fit window, excluding the beam), ``halo_q`` /
    ``halo_sigma`` (amorphous basis; ``None`` disables it), ``group_corr``.

    Returns a dict with: ``amounts``, ``fractions``, ``group_amounts`` /
    ``group_fractions``, ``halo_amount``, ``crystallinity``, ``gram`` + ``labels``,
    ``resid_frac`` (unexplained peak fraction), and ``q_fit`` / ``peaks`` / ``fit``
    / ``halo_fit`` / ``bg`` arrays for plotting.
    """
    from scipy.optimize import nnls

    names = list(candidates) if candidates is not None else list(CANDIDATES)
    q = np.asarray(q, float)
    I = np.asarray(I, float)
    ok = np.isfinite(q) & np.isfinite(I)
    q, I = q[ok], I[ok]
    if q_hi is None:
        q_hi = float(q.max())
    bg = _rolling_min_baseline(I, len(q) * bg_win_frac)
    peaks = np.clip(I - bg, 0.0, None)

    sel = (q >= q_lo) & (q <= q_hi)
    qf, pf = q[sel], peaks[sel]
    cols = [phase_ring_profile(qf, c, sigma_q) for c in names]
    has_halo = halo_q is not None and q_lo <= halo_q <= q_hi
    if has_halo:
        cols.append(np.exp(-0.5 * ((qf - halo_q) / halo_sigma) ** 2))
    A = np.column_stack(cols)
    a, _ = nnls(A, pf)
    a_phase = a[:len(names)]
    halo_amount = float(a[len(names)]) if has_halo else 0.0

    amounts = {c: float(ai) for c, ai in zip(names, a_phase)}
    tot = float(a_phase.sum())
    fractions = {c: (float(ai) / tot if tot > 0 else 0.0) for c, ai in zip(names, a_phase)}

    # normalized Gram (collinearity) of the phase fingerprints over the fit window
    R = A[:, :len(names)]
    norms = np.sqrt((R ** 2).sum(0))
    norms_safe = np.where(norms > 0, norms, 1.0)
    G = (R.T @ R) / np.outer(norms_safe, norms_safe)

    groups = _groups_by_correlation(names, G, group_corr)
    group_amounts, group_fractions = {}, {}
    for g in groups:
        key = "+".join(g)
        s = sum(amounts[c] for c in g)
        group_amounts[key] = s
        group_fractions[key] = (s / tot if tot > 0 else 0.0)

    fit = A @ a
    crys = float(np.linalg.norm(R @ a_phase))
    halo_fit = (A[:, len(names)] * halo_amount) if has_halo else np.zeros_like(qf)
    halo_energy = float(np.linalg.norm(halo_fit))
    crystallinity = crys / (crys + halo_energy) if (crys + halo_energy) > 0 else 0.0
    denom = float(np.linalg.norm(pf)) or 1.0
    resid_frac = float(np.linalg.norm(pf - fit) / denom)

    return dict(amounts=amounts, fractions=fractions,
                group_amounts=group_amounts, group_fractions=group_fractions,
                halo_amount=halo_amount, crystallinity=crystallinity,
                gram=G, labels=names, resid_frac=resid_frac,
                q_fit=qf, peaks=pf, fit=fit, halo_fit=halo_fit, bg=bg[sel])


def analyze_diffraction(mean_pat, max_pat, q_per_px, center=None,
                        candidates=None, q_beam=0.15, q_max=1.15):
    """Diffraction-only phase screen from the mean + max patterns.

    Returns a :class:`DiffractionReport`. This is the reusable core: it makes no
    assumption about how strong the signal is (weak scans just yield few rings /
    ``weak`` verdicts), so the same call runs across a whole dataset series.
    """
    mean_pat = np.asarray(mean_pat, float)
    max_pat = np.asarray(max_pat, float)
    center = _resolve_center(max_pat, center)

    # amorphous halo (FSDP) from the mean pattern
    rc, prof = _radial_median(mean_pat, center, 200, 0,
                              min(center[0], center[1],
                                  mean_pat.shape[1] - center[0],
                                  mean_pat.shape[0] - center[1]))
    q = rc * q_per_px
    good = np.isfinite(prof)
    halo_q, halo_conf = find_fsdp(q[good], prof[good],
                                  q_lo=max(q_beam, 0.10), q_hi=q_max)

    rings_q = detect_rings(max_pat, center, q_per_px, q_beam=q_beam, q_max=q_max)
    spots = detect_spots(max_pat, center, q_per_px, q_beam=q_beam, q_max=q_max)
    spq = [s[2] for s in spots]

    phases = score_phases(rings_q, spq, candidates=candidates)
    tbl = COMPOUND_RINGS
    names = list(candidates) if candidates is not None else list(CANDIDATES)

    def explained(qc):
        return any(abs(1.0 / d - qc) <= 0.045 for c in names for d, _ in tbl[c])
    unexplained = [round(1.0 / qc, 2) for qc in rings_q if not explained(qc)]

    # crystallinity = spot top-hat contrast over background noise (unitless)
    from scipy.ndimage import white_tophat
    th = white_tophat(max_pat, size=11)
    dp = max_pat.shape
    yy, xx = np.mgrid[0:dp[0], 0:dp[1]]
    rr = np.hypot(xx - center[0], yy - center[1]) * q_per_px
    ann = (rr >= q_beam) & (rr <= q_max)
    vv = th[ann]
    mad = 1.4826 * np.median(np.abs(vv - np.median(vv))) + 1e-9
    crystallinity = float((vv.max() - np.median(vv)) / mad)

    return DiffractionReport(center, q_per_px, float(halo_q), float(halo_conf),
                             [round(1.0 / qc, 3) for qc in rings_q], len(spots),
                             crystallinity, phases, unexplained)


def measure_ellipticity(max_pat, center, q_per_px, q_ring, n_wedge=36, dq=0.06):
    """Ellipticity ``eps`` and major-axis angle (deg) of a powder ring.

    Splits the ring band ``q_ring +/- dq`` into azimuthal wedges, takes the
    intensity-weighted mean radius per wedge ``r(theta)``, and fits
    ``r = r0 (1 + eps*cos 2(theta-phi))``. ``eps`` ~ (a-b)/(a+b): 0 = circular,
    a few % = mild lens/projector distortion. Returns ``(eps, angle_deg)``.
    """
    H, W = max_pat.shape
    yy, xx = np.mgrid[0:H, 0:W]
    dx, dy = xx - center[0], yy - center[1]
    r = np.hypot(dx, dy)
    th = np.arctan2(dy, dx)
    q = r * q_per_px
    band = (q > q_ring - dq) & (q < q_ring + dq)
    edges = np.linspace(-np.pi, np.pi, n_wedge + 1)
    rr, ang = [], []
    for i in range(n_wedge):
        m = band & (th >= edges[i]) & (th < edges[i + 1])
        if m.sum() < 5:
            continue
        w = np.clip(max_pat[m], 0, None) + 1e-9
        rr.append(float(np.average(r[m], weights=w)))
        ang.append(0.5 * (edges[i] + edges[i + 1]))
    if len(rr) < 8:
        return 0.0, 0.0
    rr = np.asarray(rr); ang = np.asarray(ang)
    A = np.c_[np.ones_like(ang), np.cos(2 * ang), np.sin(2 * ang)]
    r0, B, C = np.linalg.lstsq(A, rr, rcond=None)[0]
    return float(np.hypot(B, C) / max(r0, 1e-9)), float(np.degrees(0.5 * np.arctan2(C, B)))


def diagnose_cube(cube, center=None, q_per_px=None, hot_threshold=8.0):
    """Measure the corrections a scan might need, before applying any.

    Returns a dict with ``wander_px`` (std of the per-position beam center of
    mass — descan error), ``bad_pixel_frac`` (detector defects, consistently hot
    across the scan), ``ellipticity`` / ``ellipse_angle_deg`` (powder-ring
    distortion), and ``notes`` flagging which corrections are worth applying.
    Measure first, correct only what is real — over-correcting weak data (e.g.
    per-frame hot-pixel filtering) can delete genuine sparse Bragg spots.
    """
    from ..preprocess import center_of_mass, bad_pixel_map, median_pattern, clean_pattern
    from .virtual_image import center_of_mass_map
    qpp = q_per_px or cube.calibration.q_per_px
    med = clean_pattern(np.asarray(median_pattern(cube), float), hot_threshold=hot_threshold)
    if center is None:
        center = center_of_mass(med, threshold=0.3)   # on cleaned pattern (hot px would bias it)
    mx = clean_pattern(np.asarray(cube.max_dp(), float), hot_threshold=hot_threshold)
    try:
        comx, comy = center_of_mass_map(cube, center=center, normalize=True)
        wander = float(np.nanstd(np.hypot(np.asarray(comx, float), np.asarray(comy, float))))
    except Exception:
        wander = float("nan")
    try:
        bad = float(bad_pixel_map(np.asarray(cube.max_dp(), float),
                                  hot_threshold=hot_threshold).mean())
    except Exception:
        bad = float("nan")
    rings = detect_rings(mx, center, qpp)
    eps, phi = measure_ellipticity(mx, center, qpp, rings[0]) if rings else (0.0, 0.0)
    notes = []
    if np.isfinite(wander) and wander > 1.0:
        notes.append(f"beam wander {wander:.1f}px > 1 -> per-position centering may help")
    if np.isfinite(bad) and bad > 0:
        notes.append(f"{100*bad:.2f}% detector defects -> repair (consistent-hot map, keeps real spots)")
    if eps > 0.02:
        notes.append(f"ring ellipticity {100*eps:.1f}% -> elliptical-q correction improves d accuracy")
    if not notes:
        notes.append("all within tolerance -> no correction needed (avoid over-processing)")
    return dict(center=center, wander_px=wander, bad_pixel_frac=bad,
                ellipticity=eps, ellipse_angle_deg=phi,
                ring_used_A=(1.0 / rings[0] if rings else None), notes=notes)


# ------------------------------------------------------- cube: adds "where"
@dataclass
class PhaseReport:
    diffraction: DiffractionReport
    center: tuple
    material: object                   # scan-shaped bool
    location_maps: dict                # phase -> scan-shaped map (or None)
    cepstral_bands: object             # list of fluctuation maps (or None)
    fbands: tuple
    cepstral_phase_maps: dict = None   # phase -> cepstral fluctuation map at its
                                       #   diagnostic distance (or None)

    @property
    def phases(self):
        return self.diffraction.phases

    def summary(self):
        self.diffraction.summary()
        print("spatial amount (mean thickness-normalized DF over material):")
        for c in self.diffraction.phases.values():
            loc = "map" if self.location_maps.get(c.phase) is not None else "--"
            print(f"  {c.phase:7s} amount={c.amount:.3g} (ring d={c.diag_d:.2f} A) [{loc}]")


def analyze_phases(cube, q_per_px=None, center=None, candidates=None,
                   material=None, hot_threshold=8.0, with_cepstral=True,
                   fbands=((1.0, 2.0), (2.0, 3.5), (3.5, 5.5))):
    """Full phase screen from a 4D cube: identity + amount + spatial location.

    Wraps :func:`analyze_diffraction` for the "what/how-much" and adds "where":
    a thickness-normalized dark-field map per phase (at its diagnostic ring) and
    cepstral fluctuation bands for structural separation. Returns a
    :class:`PhaseReport`.
    """
    from ..preprocess import median_pattern, clean_pattern, center_of_mass
    from .virtual_image import structural_map, material_mask as _material_mask
    from .cepstral import fluctuation_multiband

    qpp = q_per_px or cube.calibration.q_per_px
    mean_pat = clean_pattern(np.asarray(median_pattern(cube), float), hot_threshold=hot_threshold)
    max_pat = clean_pattern(np.asarray(cube.max_dp(), float), hot_threshold=hot_threshold)
    if center is None:
        center = center_of_mass(mean_pat, threshold=0.3)   # cleaned first (hot px biases center)

    diff = analyze_diffraction(mean_pat, max_pat, qpp, center=center, candidates=candidates)

    scan = cube.scan_shape
    if material is None:
        try:
            material = np.asarray(_material_mask(cube, center=center), bool)
        except Exception:
            material = np.ones(scan, bool)

    Tin, Tout = 0.2 / qpp, 1.0 / qpp
    loc = {}
    for name, ev in diff.phases.items():
        if not ev.matched_d:
            loc[name] = None
            continue
        d_sel = ev.unique_d[0] if ev.unique_d else ev.matched_d[0]  # diagnostic ring
        ev.diag_d = float(d_sel)
        qc = 1.0 / d_sel
        m = np.asarray(structural_map(cube, center, (qc - 0.03) / qpp,
                                      (qc + 0.03) / qpp, Tin, Tout), float)
        loc[name] = m
        ev.amount = float(np.nanmean(m[material])) if material.any() else float(np.nanmean(m))

    cep = cep_phase = None
    if with_cepstral:
        from .cepstral import fluctuation_image
        try:
            cep = fluctuation_multiband(cube, list(fbands), qpp)
        except Exception:
            cep = None
        cep_phase = {}
        for name in diff.phases:
            d0 = PHASE_DISTANCE.get(name)
            try:
                cep_phase[name] = np.asarray(
                    fluctuation_image(cube, max(0.4, d0 - 0.35), d0 + 0.35, qpp), float)
            except Exception:
                cep_phase[name] = None
    return PhaseReport(diff, center, material, loc, cep, fbands, cep_phase)
