"""
Example: process an IN-SITU series (e.g. a temperature series of amorphous SiOx)
and track how the first-neighbour distance shifts.

    python examples/insitu_series_rdf.py path/to/folder_with_dm4s

Reproduces the original batch_dm4_to_rdf workflow with the packaged API:
locked reduction parameters across the series, per-file center/scale, then a
cross-frame track of the first RDF shell vs. temperature.
"""
import sys
import numpy as np
import fourdstem as fds


def main(folder):
    # locked reduction config -> G(r)'s stay comparable across the series
    cfg = fds.RDFConfig(composition={"Si": 1, "O": 2},
                        q_int_min=0.8, q_int_max=12.0, r_min=1.10)

    # coordinate parsed from filename (e.g. "..._450K...") by default
    series = fds.Series.from_files(folder)
    print(f"loaded {len(series)} frames; coords = {series.coordinates()}")

    # per-frame RDF (center + scale N auto per frame)
    rdfs = series.map(lambda f: fds.pattern_to_rdf(f.pattern, f.q_per_px or 1.0, cfg))

    # follow the first shell (1.3-2.2 A) across the series
    profiles = [(r.r, r.Gr) for r in rdfs]
    track = fds.track_peak(profiles, series.coordinates(), (1.3, 2.2))
    for c, p in zip(track["coord"], track["position"]):
        print(f"  coord={c!s:>7}  r1={p:.3f} A")

    # save a compact summary
    fds.save_result_npz(
        "series_rdf_summary.npz",
        coord=track["coord"], r1=track["position"], r1_amp=track["amplitude"],
        N=np.array([r.N for r in rdfs]),
    )
    print("wrote series_rdf_summary.npz")

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        fds.plot_series_waterfall(profiles, series.coordinates(), ax=ax[0])
        fds.plot_track(track, ax=ax[1], ylabel="first shell r1 (Å)")
        fig.tight_layout()
        plt.show()
    except Exception as e:
        print(f"(skipping plots: {e})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
