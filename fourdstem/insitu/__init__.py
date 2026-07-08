"""In-situ series handling: ordered frames and cross-frame feature tracking."""
from .series import Series, Frame, coordinate_from_name
from .tracking import (
    track_peak,
    track_multiple_peaks,
    integrate_region,
)

__all__ = [
    "Series", "Frame", "coordinate_from_name",
    "track_peak", "track_multiple_peaks", "integrate_region",
]
