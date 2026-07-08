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
