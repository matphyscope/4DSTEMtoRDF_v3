"""
fourdstem.io.writers
====================
Save/load results and DataCubes. Two flavours:

* ``save_result_npz`` / ``load_result_npz`` — a flat dict of named arrays
  (the natural output of an analysis, e.g. an RDF result). This mirrors the
  original batch script's per-file .npz layout.
* ``save_datacube_npz`` / ``load_datacube_npz`` — round-trip a whole DataCube
  including calibration and metadata.
"""
from __future__ import annotations
import json
import numpy as np

from .datacube import DataCube, Calibration


def save_result_npz(path, **arrays):
    """Save a flat collection of named arrays/scalars to a compressed .npz."""
    np.savez_compressed(path, **arrays)
    return path


def load_result_npz(path):
    """Load an .npz saved by :func:`save_result_npz` into a plain dict."""
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def save_datacube_npz(path, cube: DataCube):
    """Persist a DataCube (data + calibration + metadata) to a .npz."""
    cal = cube.calibration
    np.savez_compressed(
        path,
        data=cube.data,
        q_per_px=np.array(cal.q_per_px if cal.q_per_px is not None else np.nan),
        r_per_px=np.array(cal.r_per_px if cal.r_per_px is not None else np.nan),
        q_unit=cal.q_unit,
        center=np.array(cal.center if cal.center is not None else [np.nan, np.nan]),
        metadata=json.dumps(cube.metadata, default=str),
        name=cube.name or "",
    )
    return path


def load_datacube_npz(path):
    """Reconstruct a DataCube from a .npz saved by :func:`save_datacube_npz`."""
    with np.load(path, allow_pickle=True) as z:
        q_per_px = float(z["q_per_px"]) if "q_per_px" in z else None
        r_per_px = float(z["r_per_px"]) if "r_per_px" in z else None
        center = z["center"] if "center" in z else None
        if center is not None and np.all(np.isfinite(center)):
            center = (float(center[0]), float(center[1]))
        else:
            center = None
        cal = Calibration(
            q_per_px=None if q_per_px is None or np.isnan(q_per_px) else q_per_px,
            r_per_px=None if r_per_px is None or np.isnan(r_per_px) else r_per_px,
            q_unit=str(z["q_unit"]) if "q_unit" in z else "1/A",
            center=center,
        )
        meta = json.loads(str(z["metadata"])) if "metadata" in z else {}
        name = str(z["name"]) if "name" in z else None
        return DataCube(z["data"], calibration=cal, metadata=meta, name=name)
