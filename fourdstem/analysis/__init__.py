"""Analysis: azimuthal integration, virtual imaging, peaks, decomposition, RDF."""
from .azimuthal import azimuthal_integrate, azimuthal_variance, radial_profiles
from .virtual_image import (
    virtual_image,
    bright_field,
    annular_dark_field,
    center_of_mass_map,
    structural_map,
    material_mask,
    average_pattern,
    average_pattern_aligned,
)
from .peaks import (
    find_peaks_1d,
    refine_peak_parabolic,
    first_peak_position,
    peak_centroid,
    fit_gaussian_peak,
)
from .decomposition import (
    DecompositionResult,
    ProfileDecomposition,
    nmf_decompose,
    pca_decompose,
    decompose_profiles,
    cluster_cube,
    reconstruct,
)
from .interface import localize_interface
from .cepstral import (
    ewpc_pattern, quefrency_per_px, cepstral_radial_profile,
    fluctuation_image, fluctuation_multiband, ewpc_mean, ewpc_profiles,
)
from .rdf import (
    RDFConfig,
    RDFResult,
    scattering_terms,
    damping_window,
    sine_ft,
    reduce_intensity,
    reduce_profiles,
    pattern_to_rdf,
    save_rdf,
    load_rdf,
    rdf_quality,
)
from .unmix import (
    COMPOUND_SHELLS,
    synth_compound_rdf,
    build_references,
    unmix_nnls,
    reference_degeneracy,
)

__all__ = [
    "azimuthal_integrate", "azimuthal_variance", "radial_profiles",
    "virtual_image", "bright_field", "annular_dark_field", "center_of_mass_map", "structural_map", "material_mask", "average_pattern", "average_pattern_aligned",
    "find_peaks_1d", "refine_peak_parabolic", "first_peak_position", "peak_centroid",
    "fit_gaussian_peak",
    "DecompositionResult", "ProfileDecomposition", "nmf_decompose", "pca_decompose",
    "decompose_profiles", "cluster_cube", "reconstruct", "localize_interface",
    "ewpc_pattern", "quefrency_per_px", "cepstral_radial_profile",
    "fluctuation_image", "fluctuation_multiband", "ewpc_mean", "ewpc_profiles",
    "RDFConfig", "RDFResult", "scattering_terms", "damping_window", "sine_ft",
    "reduce_intensity", "reduce_profiles", "pattern_to_rdf", "save_rdf",
    "load_rdf", "rdf_quality",
    "COMPOUND_SHELLS", "synth_compound_rdf", "build_references", "unmix_nnls",
    "reference_degeneracy",
]
