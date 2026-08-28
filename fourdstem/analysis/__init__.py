"""Analysis: azimuthal integration, virtual imaging, peaks, decomposition, RDF."""
from .azimuthal import azimuthal_integrate, azimuthal_variance, radial_profiles
from .virtual_image import (
    virtual_image,
    bright_field,
    annular_dark_field,
    center_of_mass_map,
    structural_map, thickness_map,
    material_mask,
    average_pattern,
    average_pattern_aligned,
)
from .peaks import (
    find_peaks_1d,
    refine_peak_parabolic,
    first_peak_position,
    find_fsdp,
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
from .phases import (
    CANDIDATES, PHASE_DISTANCE, PhaseEvidence, DiffractionReport,
    detect_rings, detect_spots, score_phases, analyze_diffraction,
    phase_ring_profile, decompose_fractions,
    measure_ellipticity, diagnose_cube,
    PhaseReport, analyze_phases,
)
from .indexing import (
    index_gvectors, index_pattern, spots_to_gvectors, reflections as phase_reflections,
    crystallinity_map, label_grains, grain_patterns, index_grains,
    seed_positions, index_seeds,
)
from .classify import radial_stack, classify_pixels, PixelClassification
from .detectors import (strict_vacuum_mask, peak_above_flank, significance, detector_map, amorphous_halo_peaks)
from .unmix import (
    COMPOUND_SHELLS,
    synth_compound_rdf,
    build_references,
    unmix_nnls,
    COMPOUND_RINGS,
    SUBSTRATE_RINGS,
    ALL_RINGS,
    compound_ring_q,
    match_rings,
    synth_compound_iq,
    reference_degeneracy,
)

__all__ = [
    "azimuthal_integrate", "azimuthal_variance", "radial_profiles",
    "virtual_image", "bright_field", "annular_dark_field", "center_of_mass_map", "structural_map", "thickness_map", "material_mask", "average_pattern", "average_pattern_aligned",
    "find_peaks_1d", "refine_peak_parabolic", "first_peak_position", "peak_centroid", "find_fsdp",
    "fit_gaussian_peak",
    "DecompositionResult", "ProfileDecomposition", "nmf_decompose", "pca_decompose",
    "decompose_profiles", "cluster_cube", "reconstruct", "localize_interface",
    "ewpc_pattern", "quefrency_per_px", "cepstral_radial_profile",
    "fluctuation_image", "fluctuation_multiband", "ewpc_mean", "ewpc_profiles",
    "RDFConfig", "RDFResult", "scattering_terms", "damping_window", "sine_ft",
    "reduce_intensity", "reduce_profiles", "pattern_to_rdf", "save_rdf",
    "load_rdf", "rdf_quality",
    "COMPOUND_SHELLS", "synth_compound_rdf", "build_references", "unmix_nnls", "COMPOUND_RINGS", "SUBSTRATE_RINGS", "ALL_RINGS", "compound_ring_q", "match_rings", "synth_compound_iq",
    "reference_degeneracy",
    "CANDIDATES", "PHASE_DISTANCE", "PhaseEvidence", "DiffractionReport",
    "detect_rings", "detect_spots", "score_phases", "analyze_diffraction",
    "phase_ring_profile", "decompose_fractions",
    "index_gvectors", "index_pattern", "spots_to_gvectors", "phase_reflections",
    "crystallinity_map", "label_grains", "grain_patterns", "index_grains",
    "seed_positions", "index_seeds",
    "radial_stack", "classify_pixels", "PixelClassification",
    "strict_vacuum_mask", "peak_above_flank", "significance", "detector_map", "amorphous_halo_peaks",
    "measure_ellipticity", "diagnose_cube",
    "PhaseReport", "analyze_phases",
]
