"""
Example: analyze a SINGLE 4D-STEM dataset.

    python examples/single_dataset.py path/to/scan.dm4

Shows the core single-dataset workflow:
    load -> mean pattern -> beam center -> masks -> I(q) -> G(r) -> virtual images
"""
import sys
import fourdstem as fds


def main(path):
    # 1) load into a DataCube (2D/3D/4D all handled the same way)
    cube = fds.load(path)
    print(cube)
    q_per_px = cube.calibration.q_per_px or 1.0

    # 2) collapse to a mean diffraction pattern
    pattern = fds.to_pattern(cube)

    # 3) mask the beam stopper and find the center by Friedel symmetry
    stopper = fds.beam_stopper_mask(pattern)          # auto-detect
    (cx, cy), fried = fds.find_center(pattern, stopper)
    print(f"center=({cx:.1f}, {cy:.1f})  friedel={fried:.2f}")
    cube.with_center(cx, cy)

    # 4) optional: detect Bragg spots and mask them out (keep the diffuse halo)
    bragg = fds.bragg_peak_mask(pattern, center=(cx, cy), sigma=5)
    mask = fds.combine_masks(stopper, bragg)
    print(f"masked pixels: {mask.sum()}  (stopper+Bragg)")

    # 5) azimuthal profile I(q)
    q, Iq = fds.azimuthal_integrate(pattern, (cx, cy), q_per_px, mask)

    # 6) reduced density function G(r)
    cfg = fds.RDFConfig(composition={"Si": 1, "O": 2}, q_int_min=0.8, q_int_max=12.0)
    rdf = fds.pattern_to_rdf(pattern, q_per_px, cfg, center=(cx, cy), mask=bragg)
    r1, _ = fds.first_peak_position(rdf.r, rdf.Gr, 1.3, 2.2)
    print(f"N={rdf.N:.3g}  first-shell r1={r1:.2f} Å")

    # 7) virtual images (only meaningful for a 4D scan)
    if cube.ndim == 4:
        bf = fds.bright_field(cube, center=(cx, cy))
        adf = fds.annular_dark_field(cube, center=(cx, cy))
        print(f"virtual BF {bf.shape}, ADF {adf.shape}")

    # 8) plot (needs matplotlib)
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(13, 4))
        fds.show_pattern(pattern, center=(cx, cy), mask=mask, ax=ax[0])
        fds.plot_profile(q, Iq, ax=ax[1], ylabel="I(q)")
        fds.plot_rdf(rdf, ax=ax[2])
        fig.tight_layout()
        plt.show()
    except Exception as e:
        print(f"(skipping plots: {e})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
