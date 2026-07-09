"""
fourdstem
=========
A general-purpose 4D-STEM processing and analysis toolkit — conversion plus
analysis (RDF/PDF, azimuthal integration, virtual imaging, NMF/PCA
decomposition, Bragg-peak detection & masking) built to work on a *single*
4D-STEM dataset or a whole *in-situ series*.

Layout (each category is its own subpackage of small, composable functions):

    fourdstem.io          load/save, the DataCube container
    fourdstem.preprocess  calibration, beam centering, masks, transforms
    fourdstem.analysis    azimuthal I(q), virtual images, peaks, NMF/PCA, RDF
    fourdstem.insitu      ordered series + cross-frame feature tracking
    fourdstem.viz         matplotlib helpers
    fourdstem.utils       shared helpers

Quick start
-----------
    import fourdstem as fds

    cube = fds.load("scan.dm4")                 # -> DataCube
    pattern = fds.to_pattern(cube)              # mean diffraction pattern
    result = fds.pattern_to_rdf(pattern, cube.calibration.q_per_px)
    fds.plot_rdf(result)

    # in-situ temperature series -> track first-shell distance
    series = fds.Series.from_files("data/*.dm4")
    rdfs = series.map(lambda f: fds.pattern_to_rdf(f.pattern, f.q_per_px))
    profiles = [(r.r, r.Gr) for r in rdfs]
    track = fds.track_peak(profiles, series.coordinates(), (1.3, 2.2))
"""
from __future__ import annotations

__version__ = "0.1.0"

# -- io ---------------------------------------------------------------------
from .io import (
    DataCube, Calibration,
    load, load_dm4, load_dm3, load_generic, from_array, unit_to_inv_angstrom,
    mean_pattern_lazy,
    save_result_npz, load_result_npz, save_datacube_npz, load_datacube_npz,
)

# -- preprocess -------------------------------------------------------------
from .preprocess import (
    q_per_px_from_ring, pixels_to_q, q_to_pixels, q_axis, max_q_for_center,
    remove_hot_pixels, remove_dead_pixels, median_denoise, clean_pattern,
    center_of_mass, friedel_correlation, find_center_friedel, find_center,
    beam_stopper_mask, bragg_peak_mask, detect_bragg_peaks, combine_masks,
    disk_mask, annular_mask, wedge_mask,
    to_pattern, crop_detector, bin_detector, bin_cube_detector, polar_transform,
)

# -- analysis ---------------------------------------------------------------
from .analysis import (
    azimuthal_integrate, azimuthal_variance,
    virtual_image, bright_field, annular_dark_field, center_of_mass_map,
    structural_map, average_pattern,
    find_peaks_1d, refine_peak_parabolic, first_peak_position, peak_centroid,
    fit_gaussian_peak,
    DecompositionResult, ProfileDecomposition, nmf_decompose, pca_decompose,
    decompose_profiles, reconstruct,
    RDFConfig, RDFResult, scattering_terms, damping_window, sine_ft,
    reduce_intensity, pattern_to_rdf, save_rdf, load_rdf,
)

# -- insitu -----------------------------------------------------------------
from .insitu import (
    Series, Frame, coordinate_from_name,
    track_peak, track_multiple_peaks, integrate_region,
)

# -- viz --------------------------------------------------------------------
from .viz import (
    show_pattern, plot_profile, plot_rdf, plot_series_waterfall,
    plot_components, plot_track, plot_fractions,
)

# -- utils ------------------------------------------------------------------
from .utils import parallel_map, progress_iter

__all__ = [
    "__version__",
    # io
    "DataCube", "Calibration", "load", "load_dm4", "load_dm3", "load_generic",
    "from_array", "unit_to_inv_angstrom", "mean_pattern_lazy",
    "save_result_npz", "load_result_npz",
    "save_datacube_npz", "load_datacube_npz",
    # preprocess
    "q_per_px_from_ring", "pixels_to_q", "q_to_pixels", "q_axis",
    "max_q_for_center", "remove_hot_pixels", "remove_dead_pixels",
    "median_denoise", "clean_pattern", "center_of_mass", "friedel_correlation",
    "find_center_friedel", "find_center", "beam_stopper_mask", "bragg_peak_mask",
    "detect_bragg_peaks", "combine_masks", "disk_mask", "annular_mask",
    "wedge_mask", "to_pattern", "crop_detector", "bin_detector",
    "bin_cube_detector", "polar_transform",
    # analysis
    "azimuthal_integrate", "azimuthal_variance", "virtual_image", "bright_field",
    "annular_dark_field", "center_of_mass_map", "structural_map", "average_pattern", "find_peaks_1d",
    "refine_peak_parabolic", "first_peak_position", "peak_centroid",
    "fit_gaussian_peak", "DecompositionResult", "ProfileDecomposition",
    "nmf_decompose", "pca_decompose", "decompose_profiles", "reconstruct",
    "RDFConfig", "RDFResult", "scattering_terms", "damping_window", "sine_ft",
    "reduce_intensity", "pattern_to_rdf", "save_rdf", "load_rdf",
    # insitu
    "Series", "Frame", "coordinate_from_name", "track_peak",
    "track_multiple_peaks", "integrate_region",
    # viz
    "show_pattern", "plot_profile", "plot_rdf", "plot_series_waterfall",
    "plot_components", "plot_track", "plot_fractions",
    # utils
    "parallel_map", "progress_iter",
]
