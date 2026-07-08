"""Visualization helpers (matplotlib, imported lazily)."""
from .plots import (
    show_pattern,
    plot_profile,
    plot_rdf,
    plot_series_waterfall,
    plot_components,
    plot_track,
)

__all__ = [
    "show_pattern", "plot_profile", "plot_rdf", "plot_series_waterfall",
    "plot_components", "plot_track",
]
