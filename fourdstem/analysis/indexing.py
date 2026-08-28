"""
fourdstem.analysis.indexing
============================
Confirm a phase by **indexing** a single-grain diffraction pattern, not just by
matching one ring's |q|. Detecting a spot at a d-spacing only a candidate can
produce is a *prediction*; a phase is *confirmed* only when a set of spots is
self-consistent with one crystal lattice at one zone axis — their |g| ratios AND
the angles between them fit a single reciprocal lattice.

This module builds each candidate's reciprocal lattice from its unit cell (a
Cartesian reciprocal basis, so cubic / hexagonal / monoclinic all use one path),
then tries to index a measured set of g-vectors: pick two non-collinear spots as
a basis, assign them reflections with matching |g| and matching mutual angle,
take the Weiss zone axis of that pair, and count how many of the remaining spots
fall on a zone reflection at the right |g| and in-plane angle. The best
assignment gives a score (fraction of spots indexed) and the zone axis.

Use :func:`index_pattern` on a grain-averaged pattern (see the grain-first flow
in :mod:`fourdstem.analysis.phases`): index the few grains, not every pixel.
"""
from __future__ import annotations
from functools import lru_cache
import numpy as np

# Unit cells (A, deg) + Bravais centering for the systematic-absence rule.
LATTICE = {
    "LiF":    dict(a=4.0263, b=4.0263, c=4.0263, al=90, be=90,     ga=90,  centering="F"),
    "Li2O":   dict(a=4.6190, b=4.6190, c=4.6190, al=90, be=90,     ga=90,  centering="F"),
    "Li2S":   dict(a=5.7150, b=5.7150, c=5.7150, al=90, be=90,     ga=90,  centering="F"),
    "Li3N":   dict(a=3.6480, b=3.6480, c=3.8750, al=90, be=90,     ga=120, centering="P"),
    "Li2CO3": dict(a=8.3590, b=4.9770, c=6.1940, al=90, be=114.72, ga=90,  centering="C"),
}

# Bravais-lattice reflection conditions (dominant systematic absences).
CENTERING_RULE = {
    "P": lambda h, k, l: True,
    "F": lambda h, k, l: (h % 2 == k % 2 == l % 2),
    "I": lambda h, k, l: (h + k + l) % 2 == 0,
    "C": lambda h, k, l: (h + k) % 2 == 0,
}


def _recip_basis(p):
    """Cartesian reciprocal basis rows (a*, b*, c*) in 1/A for a unit cell."""
    a, b, c = p["a"], p["b"], p["c"]
    al, be, ga = np.radians([p["al"], p["be"], p["ga"]])
    av = np.array([a, 0.0, 0.0])
    bv = np.array([b * np.cos(ga), b * np.sin(ga), 0.0])
    cx = c * np.cos(be)
    cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
    cz = np.sqrt(max(c * c - cx * cx - cy * cy, 0.0))
    cv = np.array([cx, cy, cz])
    vol = np.dot(av, np.cross(bv, cv))
    return np.array([np.cross(bv, cv), np.cross(cv, av), np.cross(av, bv)]) / vol


@lru_cache(maxsize=None)
def _all_reflections(phase, hmax):
    """All allowed reflections of ``phase`` up to ``hmax`` (cached), sorted by |g|."""
    p = LATTICE[phase]
    B = _recip_basis(p)
    rule = CENTERING_RULE[p["centering"]]
    out = []
    for h in range(-hmax, hmax + 1):
        for k in range(-hmax, hmax + 1):
            for l in range(-hmax, hmax + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                if not rule(h, k, l):
                    continue
                g = h * B[0] + k * B[1] + l * B[2]
                gm = float(np.linalg.norm(g))
                if gm > 0:
                    out.append(((h, k, l), g, gm))
    out.sort(key=lambda r: r[2])
    return out


def reflections(phase, d_min, hmax=6):
    """Allowed reflections of ``phase`` down to ``d_min`` (A).

    Returns a list of ``(hkl_tuple, g_cartesian(3,), |g|)`` with ``|g| = 1/d``,
    sorted by |g|. Backed by a cached full-list generation.
    """
    g_max = 1.0 / d_min
    return [r for r in _all_reflections(phase, hmax) if r[2] <= g_max]


def _zone_completeness(zone_refl, th_ref, gm, gang, i0, mirror, tol_g, tol_r, gmax):
    """Fraction of predicted in-zone reflections (|g| <= gmax) that are observed."""
    n_pred = n_seen = 0
    for r, thp in zip(zone_refl, th_ref):
        if r[2] > gmax + tol_g:
            continue
        n_pred += 1
        for kk in range(len(gm)):
            if abs(gm[kk] - r[2]) > tol_g:
                continue
            th_meas = mirror * (gang[kk] - gang[i0])
            if abs(((thp - th_meas + np.pi) % (2 * np.pi)) - np.pi) <= tol_r:
                n_seen += 1
                break
    return n_seen / max(n_pred, 1)


def index_gvectors(gs, phase, tol_g=0.03, tol_ang=5.0, min_spots=3,
                   max_spots=30, n_basis=6):
    """Index measured g-vectors ``gs`` (N,2 in 1/A, relative to the beam) as ``phase``.

    Finds the single-zone assignment that indexes the most spots: two
    non-collinear spots (from the ``n_basis`` strongest) are matched to
    reflections with the right |g| and mutual angle, their Weiss zone axis is
    taken, and the remaining spots are checked against that zone's reflections
    (|g| within ``tol_g``, in-plane angle within ``tol_ang`` deg; both handedness
    signs tried). Completeness is scored once for the winning assignment. Returns
    ``{phase, n_matched, n_total, score, completeness, zone, residual_deg, basis,
    mirror}`` or ``None`` if fewer than ``min_spots`` spots.
    """
    gs = np.asarray(gs, float)
    n = len(gs)
    if n < min_spots:
        return None
    gm = np.hypot(gs[:, 0], gs[:, 1])
    gang = np.arctan2(gs[:, 1], gs[:, 0])
    if n > max_spots:                                  # keep the strongest (largest |g|)
        keep = np.argsort(-gm)[:max_spots]
        gs, gm, gang, n = gs[keep], gm[keep], gang[keep], max_spots
    gmax = float(gm.max())
    refl = reflections(phase, d_min=1.0 / (gmax + tol_g))
    if not refl:
        return None
    cand = [[r for r in refl if abs(r[2] - gm[i]) <= tol_g] for i in range(n)]
    order = np.argsort(-gm)
    basis_pool = order[:max(n_basis, 2)]
    tol_r = np.radians(tol_ang)
    best = None
    for a in range(len(basis_pool)):
        i0 = basis_pool[a]
        if not cand[i0]:
            continue
        for b in range(a + 1, len(basis_pool)):
            i1 = basis_pool[b]
            if not cand[i1]:
                continue
            dmeas = gang[i1] - gang[i0]
            da = abs(((dmeas + np.pi) % (2 * np.pi)) - np.pi)
            if da < np.radians(15) or da > np.radians(165):     # near-collinear
                continue
            for r0 in cand[i0]:
                for r1 in cand[i1]:
                    cser = np.clip(np.dot(r0[1], r1[1]) / (r0[2] * r1[2]), -1, 1)
                    if abs(np.arccos(cser) - abs(dmeas)) > tol_r:
                        continue
                    zone = np.cross(r0[0], r1[0])
                    if not np.any(zone):
                        continue
                    w = np.cross(r0[1], r1[1])
                    if np.linalg.norm(w) < 1e-9:
                        continue
                    u1 = r0[1] / r0[2]
                    u2 = np.cross(w, r0[1])
                    u2 /= np.linalg.norm(u2)
                    zone_refl = [r for r in refl if abs(np.dot(r[0], zone)) < 1e-6]
                    th_ref = [np.arctan2(np.dot(r[1], u2), np.dot(r[1], u1)) for r in zone_refl]
                    for mirror in (1, -1):
                        matched, res = 0, 0.0
                        for kk in range(n):
                            th_meas = mirror * (gang[kk] - gang[i0])
                            for r, thp in zip(zone_refl, th_ref):
                                if abs(r[2] - gm[kk]) > tol_g:
                                    continue
                                if abs(((thp - th_meas + np.pi) % (2 * np.pi)) - np.pi) <= tol_r:
                                    matched += 1
                                    res += abs(((thp - th_meas + np.pi) % (2 * np.pi)) - np.pi)
                                    break
                        if matched and (best is None or matched > best["n_matched"]
                                        or (matched == best["n_matched"] and res < best["_res"])):
                            best = dict(phase=phase, n_matched=int(matched), n_total=int(n),
                                        score=matched / n, zone=tuple(int(z) for z in zone),
                                        residual_deg=float(np.degrees(res / max(matched, 1))),
                                        basis=(tuple(r0[0]), tuple(r1[0])), mirror=int(mirror),
                                        _res=res, _zr=zone_refl, _tr=th_ref, _i0=int(i0))
    if best is None:
        return None
    best["completeness"] = float(_zone_completeness(
        best.pop("_zr"), best.pop("_tr"), gm, gang, best.pop("_i0"),
        best["mirror"], tol_g, tol_r, gmax))
    best.pop("_res", None)
    return best


def index_pattern(gs, candidates=None, tol_g=0.03, tol_ang=5.0, min_spots=3,
                  min_score=0.6, min_complete=0.6):
    """Index a spot set against every candidate; return the ranked results.

    ``gs`` are measured g-vectors (N,2, 1/A, relative to the beam center). Returns
    ``(best, all_results)`` where ``all_results`` is the per-phase best dict sorted
    by (n_matched, -residual). ``best['indexed']`` is True when the top phase
    reaches ``min_score`` with at least ``min_spots`` indexed spots — that is the
    'confirmed by indexing' tier, distinct from a mere |q| position match.
    """
    names = list(candidates) if candidates is not None else list(LATTICE.keys())
    results = []
    for c in names:
        r = index_gvectors(gs, c, tol_g=tol_g, tol_ang=tol_ang, min_spots=min_spots)
        if r is not None:
            cov, comp = r["score"], r.get("completeness", 0.0)
            r["f1"] = (2 * cov * comp / (cov + comp)) if (cov + comp) > 0 else 0.0
            results.append(r)
    # rank by F1 of coverage (observed spots indexed) and completeness (predicted
    # reflections seen) so a denser over-fitting lattice cannot win on coverage alone
    results.sort(key=lambda r: (-r["f1"], -r["n_matched"], r["residual_deg"]))
    best = None
    if results:
        top = dict(results[0])
        top["indexed"] = bool(top["score"] >= min_score and top["n_matched"] >= min_spots
                              and top.get("completeness", 0.0) >= min_complete)
        best = top
    return best, results


def crystallinity_map(cube, center=None, q_per_px=None, q_beam=0.20, q_max=1.15,
                      chunk=512, min_bin=16):
    """Per-scan-position 'spottiness': sharp Bragg spots vs smooth ring/halo.

    Measured as the **azimuthal** fluctuation *within each radius*, not across
    radii: a single-crystal grain concentrates intensity into discrete spots (a
    ring at one radius is bright at a few angles and dark elsewhere -> high
    azimuthal variance), while an amorphous halo or a uniform powder ring is
    azimuthally flat. At each integer radius in the annulus the Poisson-corrected
    normalized azimuthal variance ``(var - mean) / mean^2`` is formed (Poisson
    subtraction and the ``mean^2`` denominator make it independent of dose /
    thickness), and the map is the max over radii. This deliberately does NOT
    track brightness/thickness — that was the failure of a radius-mixing metric.

    Returns a scan map (high = spotty/crystalline). Radii with fewer than
    ``min_bin`` pixels are skipped; computed in chunks to stay light.
    """
    from ..preprocess.masks import annular_mask
    from .virtual_image import _resolve_center
    center = _resolve_center(cube, center)
    if q_per_px is None:
        q_per_px = cube.calibration.q_per_px
    dp = cube.dp_shape
    H, W = dp
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.hypot(xx - center[0], yy - center[1])
    ann = np.asarray(annular_mask(dp, center, q_beam / q_per_px, q_max / q_per_px), bool)
    idx = np.where(ann.ravel())[0]
    rbin = r.ravel()[idx].astype(int)
    bins = [b for b in np.unique(rbin) if (rbin == b).sum() >= min_bin]
    bin_cols = [np.where(rbin == b)[0] for b in bins]
    flat = cube._flat_patterns()
    N = flat.shape[0]
    flat2 = flat.reshape(N, -1)                        # keep dtype (no full float64 copy)
    out = np.zeros(N, float)
    for s in range(0, N, chunk):
        blk = np.asarray(flat2[s:s + chunk][:, idx], float)   # cast only this chunk
        best = np.zeros(blk.shape[0])
        for cols in bin_cols:
            sub = blk[:, cols]
            m = sub.mean(1)
            v = sub.var(1)
            cov = (v - m) / (m * m + 1e-9)             # Poisson-corrected, dose-independent
            best = np.maximum(best, cov)
        out[s:s + blk.shape[0]] = best
    scan = cube.scan_shape
    return out.reshape(scan) if scan else out


def label_grains(cryst_map, mask=None, threshold_pctl=90.0, min_size=4):
    """Label crystalline grains = connected components of high-spottiness positions.

    Positions with ``cryst_map`` above its ``threshold_pctl`` percentile (within
    ``mask`` if given) are grouped into spatially-connected grains; components
    smaller than ``min_size`` are dropped. Returns an int label map (0 = none) and
    the number of grains. Grouping crystalline pixels into grains is what makes the
    grain-first flow cheap: index the few grain averages, not every pixel.
    """
    from scipy.ndimage import label
    cm = np.asarray(cryst_map, float)
    m = np.ones_like(cm, bool) if mask is None else np.asarray(mask, bool)
    vals = cm[m]
    if vals.size == 0:
        return np.zeros_like(cm, int), 0
    thr = np.percentile(vals, threshold_pctl)
    seed = m & (cm > thr)
    lab, nlab = label(seed)
    out = np.zeros_like(lab)
    g = 0
    for i in range(1, nlab + 1):
        if (lab == i).sum() >= min_size:
            g += 1
            out[lab == i] = g
    return out, g


def grain_patterns(cube, labels):
    """Mean diffraction pattern per grain label (1..G). Returns ``{label: pattern}``."""
    from .virtual_image import average_pattern
    labels = np.asarray(labels, int)
    out = {}
    for g in range(1, int(labels.max()) + 1):
        m = labels == g
        if m.any():
            out[g] = np.asarray(average_pattern(cube, m), float)
    return out


def index_grains(cube, center=None, q_per_px=None, cryst_map=None, mask=None,
                 candidates=None, threshold_pctl=90.0, min_size=4,
                 spot_kwargs=None, index_kwargs=None):
    """Grain-first indexing: find grains, average each, detect spots, index them.

    Ties the flow together — ``crystallinity_map`` (unless ``cryst_map`` given) ->
    ``label_grains`` -> ``grain_patterns`` -> spot detection -> :func:`index_pattern`.
    Returns ``(grains, labels)`` where ``grains`` is a list of dicts per grain
    ``{label, size, n_spots, best, results}`` (``best`` is the indexing result, with
    ``best['indexed']`` the confirmed-by-indexing flag), and ``labels`` is the grain
    map. Indexes only the few grain averages, not every pixel.
    """
    from .phases import detect_spots
    from .virtual_image import _resolve_center
    center = _resolve_center(cube, center)
    if q_per_px is None:
        q_per_px = cube.calibration.q_per_px
    if cryst_map is None:
        cryst_map = crystallinity_map(cube, center=center, q_per_px=q_per_px)
    labels, ng = label_grains(cryst_map, mask=mask, threshold_pctl=threshold_pctl,
                              min_size=min_size)
    pats = grain_patterns(cube, labels)
    spot_kwargs = spot_kwargs or {}
    index_kwargs = index_kwargs or {}
    grains = []
    for g, pat in pats.items():
        spots = detect_spots(pat, center, q_per_px, **spot_kwargs)
        gs = spots_to_gvectors(spots, center, q_per_px)
        best, results = (None, [])
        if len(gs) >= index_kwargs.get("min_spots", 3):
            best, results = index_pattern(gs, candidates=candidates, **index_kwargs)
        grains.append(dict(label=g, size=int((labels == g).sum()),
                           n_spots=len(gs), best=best, results=results))
    return grains, labels


def spots_to_gvectors(spots, center, q_per_px):
    """Convert detected spots ``(x, y, q)`` (px + 1/A) to g-vectors (N,2 in 1/A).

    Uses the pixel positions and the calibration so the vectors carry both |g|
    and direction (angle), which the indexer needs — a radial |q| list alone
    cannot be indexed.
    """
    out = []
    cx, cy = center
    for s in spots:
        x, y = s[0], s[1]
        out.append(((x - cx) * q_per_px, (y - cy) * q_per_px))
    return np.asarray(out, float)
