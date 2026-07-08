"""
fourdstem.io.datacube
=====================
The central data container. A ``DataCube`` wraps a 4D-STEM dataset with shape
``(R_Ny, R_Nx, Q_Ny, Q_Nx)`` — two real-space (scan) axes and two
reciprocal-space (detector) axes — plus calibration and metadata.

The class is deliberately thin: it stores the array and calibration, and offers
a handful of convenience reductions (mean/max diffraction pattern, virtual
images). Heavy analysis lives in ``fourdstem.analysis``; this object is just the
thing they all operate on.

It also accepts lower-dimensional data so a *single* diffraction pattern (2D) or
a *stack* of patterns (3D, e.g. an in-situ time series of mean patterns) can be
carried through the same API:

    ndim 2 -> (Q_Ny, Q_Nx)                 a single diffraction pattern
    ndim 3 -> (N, Q_Ny, Q_Nx)              a stack / line scan / time series
    ndim 4 -> (R_Ny, R_Nx, Q_Ny, Q_Nx)     a full 4D-STEM scan
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Calibration:
    """Physical calibration attached to a DataCube.

    Attributes
    ----------
    q_per_px : float or None
        Reciprocal-space sampling in Å⁻¹ per detector pixel.
    r_per_px : float or None
        Real-space scan step in Å (or nm) per scan pixel.
    q_unit : str
        Human-readable unit string, e.g. ``"1/A"``.
    center : tuple or None
        Beam center ``(cx, cy)`` in detector pixels, if known.
    """
    q_per_px: float | None = None
    r_per_px: float | None = None
    q_unit: str = "1/A"
    center: tuple | None = None
    extra: dict = field(default_factory=dict)


class DataCube:
    """A 4D-STEM (or lower-dimensional) dataset with calibration and metadata."""

    def __init__(self, data, calibration=None, metadata=None, name=None):
        data = np.asarray(data)
        if data.ndim not in (2, 3, 4):
            raise ValueError(f"DataCube expects 2D/3D/4D data, got ndim={data.ndim}")
        self.data = data
        self.calibration = calibration or Calibration()
        self.metadata = metadata or {}
        self.name = name

    # -- shape helpers ------------------------------------------------------
    @property
    def ndim(self):
        return self.data.ndim

    @property
    def shape(self):
        return self.data.shape

    @property
    def dp_shape(self):
        """Detector (diffraction) shape ``(Q_Ny, Q_Nx)``."""
        return self.data.shape[-2:]

    @property
    def scan_shape(self):
        """Real-space scan shape; ``()`` for 2D, ``(N,)`` for 3D, ``(Ry, Rx)`` for 4D."""
        return self.data.shape[:-2]

    @property
    def n_patterns(self):
        """Number of diffraction patterns in the cube."""
        return int(np.prod(self.scan_shape)) if self.scan_shape else 1

    # -- reductions ---------------------------------------------------------
    def _flat_patterns(self):
        """View the data as ``(n_patterns, Q_Ny, Q_Nx)``."""
        qy, qx = self.dp_shape
        if self.ndim == 2:
            return self.data[None]
        return self.data.reshape(-1, qy, qx)

    def mean_dp(self, dtype=np.float64):
        """Mean diffraction pattern over all scan positions."""
        return self._flat_patterns().mean(0, dtype=dtype)

    def max_dp(self):
        """Maximum diffraction pattern (bright-field + all Bragg spots visible)."""
        return self._flat_patterns().max(0)

    def sum_dp(self, dtype=np.float64):
        """Summed diffraction pattern."""
        return self._flat_patterns().sum(0, dtype=dtype)

    def pattern_at(self, index):
        """Return one diffraction pattern.

        ``index`` may be an int (into the flattened stack) or a tuple matching
        the scan shape, e.g. ``(iy, ix)`` for a 4D cube.
        """
        if isinstance(index, tuple):
            return np.asarray(self.data[index], float)
        return np.asarray(self._flat_patterns()[index], float)

    def get_virtual_image(self, mask):
        """Integrate each pattern over a detector ``mask`` -> a real-space image.

        Returns a scalar (2D input), 1D array (3D input), or 2D image (4D input).
        """
        mask = np.asarray(mask, bool)
        if mask.shape != tuple(self.dp_shape):
            raise ValueError(f"mask shape {mask.shape} != dp shape {self.dp_shape}")
        flat = self._flat_patterns()
        vals = (flat * mask).sum(axis=(1, 2))
        if self.ndim == 2:
            return float(vals[0])
        if self.ndim == 3:
            return vals
        return vals.reshape(self.scan_shape)

    def roi(self, y0, y1, x0, x1):
        """Return a new DataCube cropped to a scan ROI (4D only)."""
        if self.ndim != 4:
            raise ValueError("roi() requires a 4D DataCube")
        sub = self.data[y0:y1, x0:x1]
        return DataCube(sub, self.calibration, dict(self.metadata),
                        name=(self.name or "cube") + "_roi")

    # -- convenience --------------------------------------------------------
    def with_center(self, cx, cy):
        """Return the same cube with the beam center recorded in calibration."""
        self.calibration.center = (float(cx), float(cy))
        return self

    def __repr__(self):
        cal = self.calibration
        return (f"DataCube(name={self.name!r}, shape={self.shape}, "
                f"q_per_px={cal.q_per_px}, center={cal.center})")
