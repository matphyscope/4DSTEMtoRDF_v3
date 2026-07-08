"""
fourdstem.insitu.series
=======================
Handle an in-situ *series* — many datasets acquired while a parameter (temperature,
time, bias, dose) is varied. A :class:`Series` is an ordered collection of frames;
each frame carries a mean diffraction pattern, a scalar coordinate (e.g. T or t),
and a free-form metadata dict.

Two ways to build one:

    Series.from_files(paths, ...)   load each .dm4/.emd, reduce to a mean pattern
    Series.from_cube(cube, axis=0)  slice an existing 3D/4D cube into frames

Once built you can:

    stack_patterns()   -> 3D array (n_frames, Qy, Qx) for NMF/PCA
    map(func)          apply an analysis to every frame -> list of results
    coordinates()      -> the ordered coordinate array (T/t/...)

The heavy per-frame analysis (RDF, peak finding) lives elsewhere; Series just
orchestrates iteration and keeps everything ordered and labelled.
"""
from __future__ import annotations
import os
import re
import glob
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Frame:
    """One member of a series."""
    pattern: np.ndarray            # 2D mean diffraction pattern
    coord: float                   # ordering coordinate (T, t, index, ...)
    label: str = ""
    q_per_px: float | None = None
    metadata: dict = field(default_factory=dict)


def coordinate_from_name(name, pattern=None):
    """Extract a numeric coordinate from a filename.

    Defaults to matching a temperature like ``300K`` / ``_450_`` (the original
    script's convention). Pass a custom regex with one capture group to override.
    Returns ``nan`` if nothing matches.
    """
    if pattern is not None:
        m = re.search(pattern, name)
        return float(m.group(1)) if m else np.nan
    m = (re.search(r"(\d{2,4})\s*[kK](?![a-zA-Z])", name)
         or re.search(r"[_-](\d{2,4})[_-]", name))
    return float(m.group(1)) if m else np.nan


class Series:
    """An ordered in-situ series of diffraction frames."""

    def __init__(self, frames):
        self.frames = list(frames)
        self.sort()

    # -- construction -------------------------------------------------------
    @classmethod
    def from_files(cls, paths, q_unit_hint=None, roi=None,
                   coord_regex=None, reducer="mean"):
        """Load each file, collapse to a mean pattern, wrap as a Frame.

        ``paths`` may be a glob string, a directory, or a list of paths.
        ``reducer`` in {"mean", "max"} controls how each cube collapses.
        """
        from ..io.readers import load
        from ..preprocess.transform import to_pattern

        paths = cls._resolve_paths(paths)
        frames = []
        for p in paths:
            cube = load(p, q_unit_hint)
            if reducer == "max":
                pat = cube.max_dp()
            else:
                pat = to_pattern(cube, roi)
            name = os.path.splitext(os.path.basename(p))[0]
            frames.append(Frame(
                pattern=pat,
                coord=coordinate_from_name(name, coord_regex),
                label=name,
                q_per_px=cube.calibration.q_per_px,
                metadata={"source": p, **cube.metadata},
            ))
        return cls(frames)

    @classmethod
    def from_cube(cls, cube, axis=0, coords=None, q_per_px=None):
        """Build a series by slicing a 3D/4D array or DataCube along ``axis``.

        For a 4D cube this treats one scan axis as the series axis, averaging the
        other scan axis — handy for a line-scan acquired over time.
        """
        from ..io.datacube import DataCube

        if isinstance(cube, DataCube):
            q_per_px = q_per_px or cube.calibration.q_per_px
            arr = cube.data
        else:
            arr = np.asarray(cube, float)

        if arr.ndim == 4:
            # move the chosen scan axis to front, average the other scan axis
            other = 1 - axis
            arr = np.moveaxis(arr, (axis, other), (0, 1)).mean(1)  # (n, Qy, Qx)
        elif arr.ndim == 3:
            arr = np.moveaxis(arr, axis, 0)
        else:
            raise ValueError("from_cube needs a 3D or 4D cube")

        n = arr.shape[0]
        if coords is None:
            coords = np.arange(n, dtype=float)
        frames = [Frame(pattern=np.asarray(arr[i], float), coord=float(coords[i]),
                        label=f"frame_{i:03d}", q_per_px=q_per_px)
                  for i in range(n)]
        return cls(frames)

    @staticmethod
    def _resolve_paths(paths):
        if isinstance(paths, str):
            if os.path.isdir(paths):
                found = []
                for ext in ("*.dm4", "*.dm3", "*.emd", "*.mib"):
                    found += glob.glob(os.path.join(paths, ext))
                return sorted(found)
            return sorted(glob.glob(paths))
        return list(paths)

    # -- ordering / access --------------------------------------------------
    def sort(self):
        """Sort frames by coordinate (NaNs last, preserving their order)."""
        self.frames.sort(key=lambda f: (np.isnan(f.coord), f.coord))
        return self

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        return self.frames[i]

    def __iter__(self):
        return iter(self.frames)

    def coordinates(self):
        return np.array([f.coord for f in self.frames], float)

    def labels(self):
        return [f.label for f in self.frames]

    def stack_patterns(self):
        """Return a 3D array ``(n_frames, Qy, Qx)`` (patterns must share shape)."""
        shapes = {f.pattern.shape for f in self.frames}
        if len(shapes) != 1:
            raise ValueError(f"frames have differing pattern shapes: {shapes}")
        return np.stack([f.pattern for f in self.frames], 0)

    def map(self, func):
        """Apply ``func(frame)`` to every frame, returning the list of results."""
        return [func(f) for f in self.frames]
