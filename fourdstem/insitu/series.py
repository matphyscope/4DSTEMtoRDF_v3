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
                   coord_regex=None, reducer="mean", preprocess=None):
        """Load each file, collapse to a mean pattern, wrap as a Frame.

        ``paths`` may be a glob string, a directory, or a list of paths.
        ``reducer`` in {"mean", "max"} controls how each cube collapses.
        ``preprocess`` is an optional ``f(pattern) -> pattern`` cleanup hook.
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
            if preprocess is not None:
                pat = preprocess(pat)
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
    def from_folders(cls, root, coord_regex=None, roi=None, reducer="mean",
                     q_unit_hint=None, preprocess=None, pattern="*.dm4"):
        """Build a series from SUBFOLDERS whose *name* encodes the coordinate.

        Layout expected (the common in-situ convention where each temperature is
        its own folder)::

            root/
              300K/   scan1.dm4  scan2.dm4 ...
              450K/   scan1.dm4 ...
              600K/   ...

        Each subfolder becomes one frame: its dm4(s) are loaded, collapsed to a
        mean (or max) pattern, and averaged together if there are several. The
        coordinate is parsed from the *folder* name (default: a temperature like
        ``450K``; override with ``coord_regex``).

        Parameters
        ----------
        root : str
            Parent directory containing the per-temperature subfolders.
        coord_regex : str, optional
            Regex with one capture group for the numeric coordinate.
        roi : tuple, optional
            Scan ROI passed to ``to_pattern`` for 4D cubes.
        reducer : {"mean", "max"}
            How each cube collapses to a 2D pattern.
        preprocess : callable, optional
            ``f(pattern) -> pattern`` applied to each pattern before storing
            (e.g. ``fourdstem.clean_pattern`` for hot/dead-pixel removal).
        pattern : str
            Glob for data files inside each subfolder.
        """
        import numpy as _np
        from ..io.readers import load
        from ..preprocess.transform import to_pattern

        subdirs = sorted(
            d for d in (os.path.join(root, x) for x in os.listdir(root))
            if os.path.isdir(d)
        )
        frames = []
        for d in subdirs:
            files = sorted(glob.glob(os.path.join(d, pattern)))
            if not files:
                continue
            pats, q_per_px = [], None
            for p in files:
                cube = load(p, q_unit_hint)
                q_per_px = cube.calibration.q_per_px
                pat = cube.max_dp() if reducer == "max" else to_pattern(cube, roi)
                if preprocess is not None:
                    pat = preprocess(pat)
                pats.append(pat)
            pat = pats[0] if len(pats) == 1 else _np.mean(pats, axis=0)
            name = os.path.basename(d)
            frames.append(Frame(
                pattern=pat,
                coord=coordinate_from_name(name, coord_regex),
                label=name,
                q_per_px=q_per_px,
                metadata={"folder": d, "n_files": len(files)},
            ))
        if not frames:
            raise FileNotFoundError(
                f"no '{pattern}' files found in any subfolder of {root}")
        return cls(frames)

    @classmethod
    def from_directory(cls, path, pattern="*.dm4", **kwargs):
        """Auto-detect the layout and load a series from ``path``.

        Handles the two common conventions without you having to pick:

        * **flat files** — data files named by coordinate directly inside
          ``path`` (e.g. ``0025K.dm4 … 1100K.dm4``) → dispatches to
          :meth:`from_files` (coordinate parsed from each *file* name).
        * **subfolders** — one subfolder per coordinate, each containing data
          files (e.g. ``300K/scan.dm4``) → dispatches to :meth:`from_folders`
          (coordinate parsed from each *folder* name).

        Extra keyword arguments (``coord_regex``, ``preprocess``, ``reducer``,
        ``roi``, ``q_unit_hint``) pass through to whichever loader is chosen.
        """
        direct = glob.glob(os.path.join(path, pattern))
        if direct:
            return cls.from_files(sorted(direct), **kwargs)
        has_sub = any(
            os.path.isdir(os.path.join(path, x)) and
            glob.glob(os.path.join(path, x, pattern))
            for x in os.listdir(path)
        )
        if has_sub:
            return cls.from_folders(path, pattern=pattern, **kwargs)
        raise FileNotFoundError(
            f"no '{pattern}' files found in {path} — neither directly in the "
            f"folder nor one level down in subfolders. Check the path and the "
            f"file extension (pattern={pattern!r})."
        )

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
