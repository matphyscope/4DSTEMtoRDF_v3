"""
Example: NMF decomposition of a 4D-STEM scan to separate an amorphous halo
component from crystalline Bragg components, and map where each lives.

    python examples/nmf_bragg_separation.py path/to/scan.dm4

Also shows Bragg-peak masking so the decomposition (or a later RDF) can focus on
the diffuse signal.
"""
import sys
import numpy as np
import fourdstem as fds


def main(path):
    cube = fds.load(path)
    if cube.ndim != 4:
        print(f"note: NMF loading maps are only 2D for a 4D scan; this cube is "
              f"{cube.ndim}D, so loadings will be 1D/scalar.")
    q_per_px = cube.calibration.q_per_px or 1.0

    # center + beam-stop mask from the mean pattern
    mean = cube.mean_dp()
    stopper = fds.beam_stopper_mask(mean)
    (cx, cy), _ = fds.find_center(mean, stopper)
    cube.with_center(cx, cy)

    # NMF into k components; exclude the beam stop from the factorization
    result = fds.nmf_decompose(cube, n_components=4, mask=stopper)
    print(f"NMF components: {result.components.shape}, loadings: {result.loadings.shape}")

    # inspect each component's Bragg content via azimuthal variance
    for i, comp in enumerate(result.components):
        q, var = fds.azimuthal_variance(comp, (cx, cy), q_per_px, mask=stopper)
        peak_var = float(np.nanmax(var))
        print(f"  comp {i}: peak azimuthal variance = {peak_var:.3g} "
              f"({'crystalline' if peak_var > 0.5 else 'diffuse'})")

    # Bragg peaks in the max pattern (positions, sorted by intensity)
    peaks = fds.detect_bragg_peaks(cube.max_dp(), center=(cx, cy),
                                   q_per_px=q_per_px, sigma=6)
    print(f"detected {len(peaks)} Bragg peaks; top 5:")
    for p in peaks[:5]:
        print(f"    (x={p['x']}, y={p['y']}) q={p['q']:.3f} 1/Å  I={p['intensity']:.3g}")

    try:
        import matplotlib.pyplot as plt
        fds.plot_components(result)
        plt.show()
    except Exception as e:
        print(f"(skipping plots: {e})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
