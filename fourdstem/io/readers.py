"""
fourdstem.io.readers
====================
Load microscope data into a :class:`~fourdstem.io.datacube.DataCube`.

Primary reader is ``ncempy`` (fast, no HyperSpy dependency); ``hyperspy`` is a
fallback for formats ncempy can't handle. Both are optional — a clear error is
raised listing what to install if neither is present.

Supported entry points
    load_dm4 / load_dm3   : Gatan DigitalMicrograph
    load_generic          : anything hyperspy can open (.emd, .mib, .tif, ...)
    load                  : dispatch on file extension
    from_array            : wrap an in-memory numpy array

Reciprocal calibration is auto-converted to Å⁻¹ where the unit is recognized
(1/nm or 1/Å); otherwise the raw value is kept with a warning so you can verify
against a known ring.
"""
from __future__ import annotations
import os
import warnings
import numpy as np

from .datacube import DataCube, Calibration


# ---------------------------------------------------------------------------
# unit handling
# ---------------------------------------------------------------------------
def unit_to_inv_angstrom(value, unit, hint=None):
    """Convert a reciprocal-space sampling value to Å⁻¹.

    Returns ``(value_in_invA, description)``. Recognizes 1/nm and 1/Å; anything
    else is passed through unchanged with a warning.
    """
    u = (hint or unit or "").strip().lower()
    if u in ("1/nm", "nm^-1", "nm-1", "1 / nm", "nm⁻¹"):
        return value / 10.0, "1/nm->1/A"
    if u in ("1/a", "1/å", "a^-1", "å^-1", "1 / a", "å⁻¹", "1/angstrom"):
        return value, "1/A"
    warnings.warn(
        f"unrecognized q-unit {unit!r}; using value as-is (assumed 1/A). "
        "VERIFY against a known ring or pass q_unit_hint."
    )
    return value, f"raw({unit})"


# ---------------------------------------------------------------------------
# individual readers
# ---------------------------------------------------------------------------
def _read_ncempy(path):
    from ncempy.io import dm
    d = dm.dmReader(path)
    data = np.asarray(d["data"])
    px = np.atleast_1d(d.get("pixelSize", [1.0]))
    un = d.get("pixelUnit", [""])
    un = un[-1] if isinstance(un, (list, tuple, np.ndarray)) else un
    return data, float(px[-1]), str(un), {"reader": "ncempy",
                                          "pixelSize": px.tolist()}


def _read_hyperspy(path):
    import hyperspy.api as hs
    sig = hs.load(path)
    data = np.asarray(sig.data)
    ax = sig.axes_manager[-1]
    return data, float(ax.scale), str(ax.units), {"reader": "hyperspy"}


def _finalize(data, scale, unit, meta, hint, name):
    q_per_px, conv = unit_to_inv_angstrom(scale, unit, hint)
    meta = dict(meta)
    meta.update({"unit": unit, "conv": conv, "source": name})
    cal = Calibration(q_per_px=q_per_px, q_unit="1/A", extra={"raw_unit": unit})
    return DataCube(data, calibration=cal, metadata=meta, name=name)


def load_dm4(path, q_unit_hint=None):
    """Load a Gatan .dm4 file into a DataCube (ncempy primary, hyperspy fallback)."""
    name = os.path.splitext(os.path.basename(path))[0]
    try:
        data, scale, unit, meta = _read_ncempy(path)
    except Exception as e_nc:
        try:
            data, scale, unit, meta = _read_hyperspy(path)
        except Exception as e_hs:
            raise RuntimeError(
                f"could not read {path}. Install 'ncempy' or 'hyperspy'.\n"
                f"  ncempy error: {e_nc}\n  hyperspy error: {e_hs}"
            )
    return _finalize(data, scale, unit, meta, q_unit_hint, name)


# .dm3 uses the same reader path
load_dm3 = load_dm4


def load_generic(path, q_unit_hint=None):
    """Load any format HyperSpy understands (.emd, .mib, .tif, .hspy, ...)."""
    name = os.path.splitext(os.path.basename(path))[0]
    data, scale, unit, meta = _read_hyperspy(path)
    return _finalize(data, scale, unit, meta, q_unit_hint, name)


def mean_pattern_lazy(path, q_unit_hint=None, reducer="mean", roi=None,
                      chunk_rows=1):
    """Memory-light mean/max diffraction pattern from a big 4D dm4 via memmap.

    For a 5-6 GB 4D cube (e.g. 150x150x256x256), loading the whole thing just to
    average it is wasteful and risky under parallelism. This memory-maps the file
    (ncempy) and accumulates the reduction over the scan axis in chunks, holding
    only a few detector frames in RAM at a time. It returns the same
    ``(pattern_2d, q_per_px, metadata)`` as a full load.

    Falls back to a full :func:`load` + reduction if memmap is unavailable or the
    file isn't 4D, so it never breaks — worst case it behaves like the eager path.

    reducer : {"mean", "max"}
    roi : (y0, y1, x0, x1) scan ROI, optional.
    chunk_rows : number of scan rows to read per chunk.
    """
    try:
        from ncempy.io import dm
        fdm = dm.fileDM(path)
        try:
            mm = fdm.getMemmap(0)
        except Exception:
            mm = None
        if mm is None or np.ndim(mm) != 4:
            raise RuntimeError("memmap unavailable or not 4D")

        sy, sx, dy, dx = mm.shape
        y0, y1, x0, x1 = (roi if roi is not None else (0, sy, 0, sx))
        acc = None
        count = 0
        for iy in range(y0, y1, max(1, chunk_rows)):
            iy1 = min(iy + max(1, chunk_rows), y1)
            block = np.asarray(mm[iy:iy1, x0:x1], dtype=np.float64)  # (r, cx, dy, dx)
            block = block.reshape(-1, dy, dx)
            if reducer == "max":
                b = block.max(0)
                acc = b if acc is None else np.maximum(acc, b)
            else:
                b = block.sum(0)
                acc = b if acc is None else acc + b
                count += block.shape[0]
        pattern = acc if reducer == "max" else acc / max(count, 1)

        # calibration from the file header (detector = last dimension)
        try:
            scale = float(fdm.scale[0][-1])
            unit = str(fdm.scaleUnit[0][-1])
        except Exception:
            scale, unit = 1.0, ""
        q_per_px, conv = unit_to_inv_angstrom(scale, unit, q_unit_hint)
        meta = {"reader": "ncempy-memmap", "unit": unit, "conv": conv,
                "source": os.path.splitext(os.path.basename(path))[0]}
        return pattern, q_per_px, meta
    except Exception:
        # robust fallback: full load then reduce (to_pattern is memory-safe)
        from ..preprocess.transform import to_pattern
        cube = load(path, q_unit_hint)
        pat = cube.max_dp() if reducer == "max" else to_pattern(cube, roi)
        return pat, cube.calibration.q_per_px, cube.metadata


def from_array(data, q_per_px=None, r_per_px=None, name="array", metadata=None):
    """Wrap an in-memory array (already loaded elsewhere) into a DataCube."""
    cal = Calibration(q_per_px=q_per_px, r_per_px=r_per_px)
    return DataCube(np.asarray(data), calibration=cal,
                    metadata=metadata or {}, name=name)


_DM = (".dm4", ".dm3")


def load(path, q_unit_hint=None):
    """Dispatch to the right reader based on file extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext in _DM:
        return load_dm4(path, q_unit_hint)
    if ext == ".npz":
        from .writers import load_datacube_npz
        return load_datacube_npz(path)
    return load_generic(path, q_unit_hint)
