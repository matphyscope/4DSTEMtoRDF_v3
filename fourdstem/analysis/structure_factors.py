"""Electron powder-ring intensities from crystal structure factors.

The ring *positions* in :mod:`unmix` are crystallographic d-spacings, but their
intensities there are only *rough* nominal weights. That misses two physical
effects that matter for these Li-SEI phases:

* **Z-dependent electron scattering** — Li (Z=3) scatters weakly; F/O/N/S/C
  dominate. A ring that looks strong in a nominal table can be Li-dominated and
  nearly invisible in real electron diffraction.
* **Structure-factor cancellation** — e.g. rock-salt 111 ~ (f_anion - f_Li) is
  weak while 200 ~ (f_anion + f_Li) is strong, and some reflections vanish by
  systematic absence.

This module computes the real electron powder intensity for each ring,

    I_hkl  ∝  Σ_{equivalent hkl} |F_hkl|²,
    F_hkl  =  Σ_j f_e,j(q=1/d) · exp[2πi (h x_j + k y_j + l z_j)],

using the Kirkland electron scattering factors (:func:`fourdstem.analysis.rdf._f_kirkland`,
with Li/F/S bundled) and the conventional-cell atomic positions below. Summing
|F|² over the symmetry-equivalent reflections that share a d-spacing folds in the
multiplicity automatically. Lattice parameters and atomic coordinates are from
standard structure determinations (rock-salt LiF; antifluorite Li2O/Li2S;
hexagonal P6/mmm Li3N; monoclinic C2/c Li2CO3 / zabuyelite).
"""

import numpy as np

from .rdf import _kirkland_params, _f_kirkland

__all__ = ["CRYSTALS", "electron_rings", "electron_ring_table"]


def _cubic(a):
    return (a, a, a, 90.0, 90.0, 90.0)


_FCC = [(0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)]


def _expand(reps, trans):
    """Add each centering translation in ``trans`` to every representative atom."""
    out = []
    for el, x, y, z in reps:
        for tx, ty, tz in trans:
            out.append((el, (x + tx) % 1.0, (y + ty) % 1.0, (z + tz) % 1.0))
    return out


def _c2c(reps):
    """Expand asymmetric-unit atoms by the 8 general positions of C2/c (#15)."""
    ops = [
        lambda x, y, z: (x, y, z),
        lambda x, y, z: (-x, y, -z + 0.5),
        lambda x, y, z: (-x, -y, -z),
        lambda x, y, z: (x, -y, z + 0.5),
    ]
    cent = [(0.0, 0.0, 0.0), (0.5, 0.5, 0.0)]
    out = []
    for el, x, y, z in reps:
        for op in ops:
            xx, yy, zz = op(x, y, z)
            for cx, cy, cz in cent:
                out.append((el, (xx + cx) % 1.0, (yy + cy) % 1.0, (zz + cz) % 1.0))
    uniq = []
    for a in out:
        if not any(a[0] == b[0] and abs(a[1] - b[1]) < 1e-4 and
                   abs(a[2] - b[2]) < 1e-4 and abs(a[3] - b[3]) < 1e-4 for b in uniq):
            uniq.append(a)
    return uniq


# Conventional-cell structures: lattice (a,b,c,alpha,beta,gamma) Å/deg + full atom list.
CRYSTALS = {
    "LiF":  dict(lattice=_cubic(4.0263),
                 atoms=_expand([("Li", 0, 0, 0), ("F", 0.5, 0.5, 0.5)], _FCC)),
    "Li2O": dict(lattice=_cubic(4.6114),
                 atoms=_expand([("O", 0, 0, 0),
                                ("Li", 0.25, 0.25, 0.25), ("Li", 0.75, 0.75, 0.75)], _FCC)),
    "Li2S": dict(lattice=_cubic(5.7159),
                 atoms=_expand([("S", 0, 0, 0),
                                ("Li", 0.25, 0.25, 0.25), ("Li", 0.75, 0.75, 0.75)], _FCC)),
    "Li3N": dict(lattice=(3.648, 3.648, 3.875, 90.0, 90.0, 120.0),
                 atoms=[("N", 0, 0, 0), ("Li", 0, 0, 0.5),
                        ("Li", 1.0 / 3, 2.0 / 3, 0), ("Li", 2.0 / 3, 1.0 / 3, 0)]),
    "Li2CO3": dict(lattice=(8.3593, 4.9767, 6.1975, 90.0, 114.83, 90.0),
                   atoms=_c2c([("Li", 0.1969, 0.0025, 0.3242), ("C", 0.0, 0.0683, 0.25),
                               ("O", 0.0, 0.2757, 0.25), ("O", 0.1441, 0.9376, 0.3160)])),
}


def _recip_metric(lattice):
    """Reciprocal metric tensor G* (so 1/d² = [hkl] · G* · [hkl]ᵀ)."""
    a, b, c, al, be, ga = lattice
    al, be, ga = np.radians([al, be, ga])
    G = np.array([
        [a * a, a * b * np.cos(ga), a * c * np.cos(be)],
        [a * b * np.cos(ga), b * b, b * c * np.cos(al)],
        [a * c * np.cos(be), b * c * np.cos(al), c * c],
    ])
    return np.linalg.inv(G)


def electron_rings(compound, d_min=1.0, d_max=6.0, hmax=6, min_w=0.03,
                   normalize=True):
    """Electron powder rings ``[(d Å, rel_intensity), ...]`` for one compound.

    Enumerates ``hkl`` up to ``±hmax``, computes each reflection's d-spacing and
    electron structure factor, and sums ``|F|²`` over reflections sharing a
    d-spacing (folds in multiplicity). Returns rings with ``d_min ≤ d ≤ d_max``
    and relative intensity ``≥ min_w``, normalized to the strongest = 1.0 when
    ``normalize``. Sorted by d descending (same layout as
    :data:`fourdstem.analysis.unmix.COMPOUND_RINGS`).
    """
    cr = CRYSTALS[compound]
    Gs = _recip_metric(cr["lattice"])
    atoms = cr["atoms"]
    tbl = _kirkland_params()
    fcache = {}
    refl = {}
    for h in range(-hmax, hmax + 1):
        for k in range(-hmax, hmax + 1):
            for l in range(-hmax, hmax + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                hkl = np.array([h, k, l], float)
                s2 = float(hkl @ Gs @ hkl)
                if s2 <= 0:
                    continue
                d = 1.0 / np.sqrt(s2)
                if d < d_min or d > d_max:
                    continue
                q = 1.0 / d
                F = 0.0 + 0.0j
                for el, x, y, z in atoms:
                    fe = fcache.get((el, round(q, 4)))
                    if fe is None:
                        fe = float(_f_kirkland(np.array([q]), tbl[el])[0])
                        fcache[(el, round(q, 4))] = fe
                    F += fe * np.exp(2j * np.pi * (h * x + k * y + l * z))
                I = float(abs(F) ** 2)
                key = round(d, 3)
                refl[key] = refl.get(key, 0.0) + I
    if not refl:
        return []
    imax = max(refl.values())
    rings = [(d, I / imax if normalize else I) for d, I in refl.items()]
    rings = [(d, w) for d, w in rings if w >= min_w]
    return sorted(rings, key=lambda t: -t[0])


def electron_ring_table(compounds=None, **kw):
    """`{compound: electron_rings(compound)}` for a set of candidates."""
    if compounds is None:
        compounds = list(CRYSTALS)
    return {c: electron_rings(c, **kw) for c in compounds}
