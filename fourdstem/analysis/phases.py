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


def score_phases(rings_q, spots_q, candidates=None, tol=0.045, min_unique_spots=5):
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
    """
    tbl = COMPOUND_RINGS
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
        if len(ow) == 1:
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
    mean_pat = np.asarray(median_pattern(cube), float)
    max_pat = clean_pattern(np.asarray(cube.max_dp(), float), hot_threshold=hot_threshold)
    if center is None:
        center = center_of_mass(mean_pat, threshold=0.3)

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
