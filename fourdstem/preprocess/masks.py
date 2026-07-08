"""
fourdstem.preprocess.masks
==========================
Detector-space masks. Masks are boolean arrays over the diffraction pattern;
by convention ``True`` marks pixels to EXCLUDE from analysis (a beam stopper,
a Bragg spot) OR — for virtual detectors — pixels to INCLUDE. Each function's
docstring says which.

Groups
------
Excluders (True = drop):
    beam_stopper_mask   auto/box mask of the beam-stop rod
    bragg_peak_mask     detect & mask sharp Bragg spots (keep the diffuse ring)
    combine_masks       OR several masks together

Selectors (True = keep) — "virtual detectors":
    disk_mask           bright-field disk of radius r
    annular_mask        ADF/DF annulus between r_inner and r_outer
    wedge_mask          azimuthal wedge (for anisotropy / DPC-like signals)
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import label, binary_dilation, median_filter, maximum_filter

from ..utils.helpers import radial_coordinate, angular_coordinate


# ---------------------------------------------------------------------------
# excluders
# ---------------------------------------------------------------------------
def beam_stopper_mask(img, coords=None, low_frac=0.12, dilate=6):
    """Mask the beam-stopper rod. ``True`` = stopper (exclude).

    If ``coords=(x0, y0, x1, y1)`` is given, use that fixed box (preferred, since
    the stopper is physically fixed). Otherwise auto-detect a dark, thin,
    edge-connected feature and dilate it.
    """
    img = np.asarray(img, float)
    if coords is not None:
        x0, y0, x1, y1 = coords
        m = np.zeros(img.shape, bool)
        m[y0:y1, x0:x1] = True
        return m
    lbl, n = label(img < low_frac * img.max())
    H, W = img.shape
    m = np.zeros(img.shape, bool)
    for i in range(1, n + 1):
        ys, xs = np.where(lbl == i)
        edge = xs.min() == 0 or xs.max() == W - 1 or ys.min() == 0 or ys.max() == H - 1
        if edge and (xs.max() - xs.min()) > 0.35 * W and (ys.max() - ys.min()) < 0.25 * H:
            m[ys, xs] = True
    return binary_dilation(m, iterations=dilate)


def _azimuthal_background(img, center, mask=None):
    """Back-projected isotropic (radial-mean) background about ``center``.

    Subtracting this leaves ~0 for an azimuthally-uniform ring but preserves
    localized Bragg spots — the physically correct background when the goal is
    to separate crystalline spots from an amorphous halo.
    """
    r = radial_coordinate(img.shape, center)
    rbin = np.rint(r).astype(int)
    valid = np.isfinite(img)
    if mask is not None:
        valid &= ~np.asarray(mask, bool)
    nb = rbin.max() + 1
    cnt = np.bincount(rbin[valid].ravel(), minlength=nb).astype(float)
    tot = np.bincount(rbin[valid].ravel(), weights=img[valid].ravel(), minlength=nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        prof = tot / cnt
    prof = np.nan_to_num(prof, nan=0.0)
    return prof[rbin]


def bragg_peak_mask(img, center=None, q_per_px=None, sigma=4.0, min_distance=3,
                    radius=2, exclude_center_frac=0.05, background="auto",
                    min_prominence_frac=0.02):
    """Detect sharp Bragg spots and return a mask (``True`` = Bragg spot).

    Designed for the classic problem of pulling a *diffuse/amorphous* signal out
    of a pattern contaminated by crystalline Bragg reflections (e.g. a
    nanocrystal embedded in an amorphous matrix). Apply the returned mask before
    azimuthal integration / RDF to suppress the spots.

    Method: subtract a background, then keep local maxima that stand ``sigma``
    robust-deviations above the residual AND above an absolute prominence floor.
    Each surviving peak is grown to a small disk of ``radius`` pixels.

    Parameters
    ----------
    img : 2D array
        Diffraction pattern.
    center : (cx, cy), optional
        Beam center; used for the azimuthal background and to exclude the
        central beam. Recommended.
    sigma : float
        Detection threshold in units of the residual's robust std (MAD-based).
    min_distance : int
        Minimum separation between peaks (via a maximum filter footprint).
    radius : int
        Radius (px) of the disk masked around each detected peak.
    exclude_center_frac : float
        Radius, as a fraction of the pattern half-size, around ``center`` to
        ignore (so the central beam isn't flagged).
    background : {"auto", "azimuthal", "median", "none"}
        Background model subtracted before detection. ``"auto"`` uses the
        azimuthal (radial-mean) background when ``center`` is given — which
        cancels a uniform ring — and falls back to a local median otherwise.
    min_prominence_frac : float
        Absolute floor: a peak's residual must also exceed this fraction of the
        residual's dynamic range. Rejects the tiny numeric ripples left on a
        smooth ring even when the robust std is ~0.
    """
    img = np.asarray(img, float)
    H, W = img.shape

    if background == "auto":
        background = "azimuthal" if center is not None else "median"

    if background == "azimuthal" and center is not None:
        resid = img - _azimuthal_background(img, center)
    elif background == "median":
        bg = median_filter(img, size=max(3, 2 * min_distance + 1))
        resid = img - bg
    else:
        resid = img - np.nanmedian(img)

    # robust spread (MAD -> sigma-equivalent)
    med = np.nanmedian(resid)
    mad = np.nanmedian(np.abs(resid - med)) + 1e-12
    robust_std = 1.4826 * mad
    # absolute floor tied to the ORIGINAL image's dynamic range, so a smooth
    # ring (tiny residual) can't clear it while true spots easily do.
    img_span = np.nanmax(img) - np.nanmedian(img) + 1e-12
    thresh = max(med + sigma * robust_std, med + min_prominence_frac * img_span)

    # STRICT local maxima: greater than every neighbour (center excluded). A flat
    # azimuthal ridge from an isotropic ring has none; a compact spot does.
    k = 2 * min_distance + 1
    footprint = np.ones((k, k), bool)
    footprint[min_distance, min_distance] = False
    neighbor_max = maximum_filter(resid, footprint=footprint)
    peaks = (resid > neighbor_max) & (resid > thresh)

    if center is not None:
        r = radial_coordinate(img.shape, center)
        peaks &= r > exclude_center_frac * 0.5 * min(H, W)

    if radius > 0 and peaks.any():
        struct = np.ones((2 * radius + 1, 2 * radius + 1), bool)
        peaks = binary_dilation(peaks, structure=struct)
    return peaks


def detect_bragg_peaks(img, center=None, q_per_px=None, **kwargs):
    """Return a list of detected Bragg peaks as records.

    Each record: ``dict(x, y, intensity, q)`` where ``q`` is in Å⁻¹ if
    ``center`` and ``q_per_px`` are supplied (else ``nan``). Useful for peak
    *positions* rather than just masking.
    """
    kwargs.setdefault("radius", 0)  # we want single-pixel peaks here
    mask = bragg_peak_mask(img, center=center, q_per_px=q_per_px, **kwargs)
    ys, xs = np.where(mask)
    out = []
    for x, y in zip(xs, ys):
        if center is not None and q_per_px is not None:
            cx, cy = center
            q = float(np.hypot(x - cx, y - cy) * q_per_px)
        else:
            q = float("nan")
        out.append(dict(x=int(x), y=int(y), intensity=float(img[y, x]), q=q))
    out.sort(key=lambda p: -p["intensity"])
    return out


def combine_masks(*masks):
    """Logical OR of several boolean masks (ignoring ``None``)."""
    masks = [np.asarray(m, bool) for m in masks if m is not None]
    if not masks:
        raise ValueError("combine_masks needs at least one non-None mask")
    out = masks[0].copy()
    for m in masks[1:]:
        out |= m
    return out


# ---------------------------------------------------------------------------
# selectors (virtual detectors) — True = keep
# ---------------------------------------------------------------------------
def disk_mask(shape, center, radius):
    """Filled disk (bright-field detector). ``True`` = inside radius."""
    r = radial_coordinate(shape, center)
    return r <= radius


def annular_mask(shape, center, r_inner, r_outer):
    """Annulus (ADF/DF detector). ``True`` = ``r_inner < r <= r_outer``."""
    r = radial_coordinate(shape, center)
    return (r > r_inner) & (r <= r_outer)


def wedge_mask(shape, center, theta0, theta1, r_inner=0.0, r_outer=None):
    """Azimuthal wedge between angles (radians), optionally radially bounded.

    Angles follow ``arctan2(y-cy, x-cx)``. Handles wrap-around when
    ``theta0 > theta1``.
    """
    r = radial_coordinate(shape, center)
    th = angular_coordinate(shape, center)
    if r_outer is None:
        r_outer = np.inf
    radial = (r > r_inner) & (r <= r_outer)
    if theta0 <= theta1:
        ang = (th >= theta0) & (th <= theta1)
    else:  # wrap across +/-pi
        ang = (th >= theta0) | (th <= theta1)
    return radial & ang
