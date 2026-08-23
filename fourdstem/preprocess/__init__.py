"""Preprocessing: calibration, centering, masks, geometric transforms, cleanup."""
from .calibration import (
    q_per_px_from_ring,
    pixels_to_q,
    q_to_pixels,
    q_axis,
    max_q_for_center,
)
from .clean import (
    remove_hot_pixels,
    remove_dead_pixels,
    median_denoise,
    clean_pattern,
)
from .center import (
    center_of_mass,
    friedel_correlation,
    find_center_friedel,
    find_center,
)
from .masks import (
    beam_stopper_mask,
    bragg_peak_mask,
    detect_bragg_peaks,
    combine_masks,
    disk_mask,
    annular_mask,
    wedge_mask,
)
from .transform import (
    to_pattern,
    median_pattern,
    crop_detector,
    bin_detector,
    bin_cube_detector,
    polar_transform,
)

__all__ = [
    "q_per_px_from_ring", "pixels_to_q", "q_to_pixels", "q_axis",
    "max_q_for_center",
    "remove_hot_pixels", "remove_dead_pixels", "median_denoise", "clean_pattern",
    "center_of_mass", "friedel_correlation", "find_center_friedel", "find_center",
    "beam_stopper_mask", "bragg_peak_mask", "detect_bragg_peaks", "combine_masks",
    "disk_mask", "annular_mask", "wedge_mask",
    "to_pattern", "median_pattern", "crop_detector", "bin_detector",
    "bin_cube_detector", "polar_transform",
]
