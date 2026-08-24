"""End-to-end tests on synthetic data (numpy/scipy only; sklearn optional)."""
import numpy as np
import pytest

import fourdstem as fds
from tests.synthetic import ring_pattern, bragg_pattern, scan_cube


# -- io / DataCube ----------------------------------------------------------
def test_datacube_shapes_and_reductions():
    cube = fds.from_array(scan_cube((4, 3), (64, 64)), q_per_px=0.02)
    assert cube.ndim == 4
    assert cube.scan_shape == (4, 3)
    assert cube.dp_shape == (64, 64)
    assert cube.n_patterns == 12
    assert cube.mean_dp().shape == (64, 64)
    assert cube.max_dp().shape == (64, 64)


def test_mean_pattern_lazy_fallback(tmp_path):
    # non-dm4 file: memmap path fails -> falls back to full load + reduce.
    from fourdstem.io.writers import save_datacube_npz
    cube4d = np.random.default_rng(0).integers(0, 100, (4, 5, 12, 12)).astype(np.uint16)
    dc = fds.from_array(cube4d, q_per_px=0.02)
    p = tmp_path / "c.npz"
    save_datacube_npz(str(p), dc)
    pat, q_per_px, meta = fds.mean_pattern_lazy(str(p))
    assert pat.shape == (12, 12)
    expected = cube4d.reshape(-1, 12, 12).mean(0, dtype=np.float64)
    assert np.allclose(pat, expected)
    assert q_per_px == pytest.approx(0.02)


def test_bin_cube_detector():
    cube = np.arange(2 * 3 * 8 * 8, dtype=np.uint16).reshape(2, 3, 8, 8)
    binned = fds.bin_cube_detector(cube, 4)
    assert binned.shape == (2, 3, 2, 2)
    # block-mean of the top-left 4x4 block of pattern [0,0]
    assert binned[0, 0, 0, 0] == pytest.approx(cube[0, 0, :4, :4].mean())
    # DataCube path scales q_per_px by the factor
    dc = fds.from_array(cube, q_per_px=0.01)
    dcb = fds.bin_cube_detector(dc, 4)
    assert dcb.dp_shape == (2, 2)
    assert dcb.calibration.q_per_px == pytest.approx(0.04)


def test_to_pattern_preserves_dtype_and_mean():
    # 4D uint16 cube: to_pattern must accumulate in float64 without upcasting
    # the whole cube, and return the correct mean.
    rng = np.random.default_rng(0)
    cube = rng.integers(0, 500, size=(4, 5, 16, 16), dtype=np.uint16)
    pat = fds.to_pattern(cube)
    assert pat.dtype == np.float64
    assert pat.shape == (16, 16)
    expected = cube.reshape(-1, 16, 16).mean(0, dtype=np.float64)
    assert np.allclose(pat, expected)


def test_datacube_roundtrip_npz(tmp_path):
    cube = fds.from_array(ring_pattern((48, 48)), q_per_px=0.03, name="p")
    p = tmp_path / "cube.npz"
    fds.save_datacube_npz(str(p), cube)
    back = fds.load_datacube_npz(str(p))
    assert back.calibration.q_per_px == pytest.approx(0.03)
    assert np.allclose(back.data, cube.data)


# -- centering --------------------------------------------------------------
def test_find_center_recovers_offset():
    true_c = (70.0, 55.0)
    img = ring_pattern((128, 128), center=true_c,
                       rings=((25, 6, 1.0), (45, 6, 0.6)))
    (cx, cy), fried = fds.find_center(img)
    assert abs(cx - true_c[0]) < 1.5
    assert abs(cy - true_c[1]) < 1.5
    assert fried > 0.8


def test_center_of_mass_on_central_beam():
    img = ring_pattern((64, 64), center=(40, 30), rings=(), central_beam=10)
    cx, cy = fds.center_of_mass(img, threshold=0.5)
    assert abs(cx - 40) < 1.0 and abs(cy - 30) < 1.0


# -- masks ------------------------------------------------------------------
def test_bragg_mask_flags_spots_not_halo():
    center = (64, 64)
    img = bragg_pattern((128, 128), center=center, spots=6, radius=45, amp=8)
    mask = fds.bragg_peak_mask(img, center=center, sigma=5)
    assert mask.sum() > 0
    # amorphous ring pattern should trigger far fewer detections
    halo = ring_pattern((128, 128), center=center, rings=((45, 8, 1.0),))
    mask_halo = fds.bragg_peak_mask(halo, center=center, sigma=5)
    assert mask.sum() > mask_halo.sum()


def test_detect_bragg_peaks_positions():
    center = (64, 64)
    img = bragg_pattern((128, 128), center=center, spots=6, radius=45, amp=8)
    peaks = fds.detect_bragg_peaks(img, center=center, q_per_px=0.02, sigma=5)
    assert len(peaks) >= 4
    qs = [p["q"] for p in peaks[:6]]
    assert np.median(qs) == pytest.approx(45 * 0.02, abs=0.05)


def test_average_pattern_over_scan_mask():
    cube = np.arange(4 * 5 * 6 * 6, dtype=float).reshape(4, 5, 6, 6)
    dc = fds.from_array(cube)
    mask = np.zeros((4, 5), bool)
    mask[0, 0] = True
    mask[1, 2] = True
    avg = fds.average_pattern(dc, mask)
    assert avg.shape == (6, 6)
    assert np.allclose(avg, (cube[0, 0] + cube[1, 2]) / 2)


def test_structural_map_cancels_brightness():
    # 4D: left half ring at r=8, right half ring at r=13; a per-position
    # brightness ramp. structural_map (ring/total) should reveal the structural
    # left/right split, not the brightness ramp.
    from tests.synthetic import ring_pattern
    a = ring_pattern((48, 48), rings=((8, 2, 1.0),), central_beam=2)
    b = ring_pattern((48, 48), rings=((13, 2, 1.0),), central_beam=2)
    Ry, Rx = 5, 8
    cube = np.empty((Ry, Rx, 48, 48))
    for iy in range(Ry):
        for ix in range(Rx):
            cube[iy, ix] = (1 + 4 * ix / Rx) * (a if ix < Rx // 2 else b)
    dc = fds.from_array(cube, q_per_px=1.0)
    smap = fds.structural_map(dc, center=(24, 24), r_inner=6, r_outer=10)
    # inner-ring (r=8) intensity dominates the left half -> higher structural value
    assert smap[:, :Rx // 2].mean() > smap[:, Rx // 2:].mean()


def test_virtual_detector_masks():
    shape = (64, 64)
    center = (32, 32)
    d = fds.disk_mask(shape, center, 10)
    a = fds.annular_mask(shape, center, 10, 20)
    assert d.sum() > 0 and a.sum() > 0
    assert not (d & a).any()  # disjoint


# -- azimuthal + peaks ------------------------------------------------------
def test_azimuthal_integrate_finds_ring():
    center = (64, 64)
    img = ring_pattern((128, 128), center=center, rings=((40, 5, 1.0),),
                       central_beam=0.0)
    q, Iq = fds.azimuthal_integrate(img, center, q_per_px=1.0)
    peak_idx = np.nanargmax(Iq)
    assert abs(q[peak_idx] - 40) < 2.0


def test_first_peak_position_refines():
    x = np.linspace(0, 10, 200)
    y = np.exp(-((x - 3.7) ** 2) / (2 * 0.2 ** 2))
    xp, yp = fds.first_peak_position(x, y, 3.0, 4.5)
    assert xp == pytest.approx(3.7, abs=0.05)


# -- RDF --------------------------------------------------------------------
def test_pattern_to_rdf_runs():
    center = (64, 64)
    img = ring_pattern((128, 128), center=center,
                       rings=((30, 6, 1.0), (55, 6, 0.5)))
    cfg = fds.RDFConfig(q_int_min=0.3, q_int_max=2.0, r_max=8.0)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = fds.pattern_to_rdf(img, q_per_px=0.03, cfg=cfg)
    assert res.Gr.shape == res.r.shape
    assert np.isfinite(res.N)
    assert res.center is not None


def test_reduce_intensity_raises_on_empty_window():
    # q only spans 0..0.15 but the window wants 0.8..12 -> must fail loudly
    q = np.linspace(0.0, 0.15, 120)
    Iq = np.exp(-q / 0.03)
    cfg = fds.RDFConfig(q_int_min=0.8, q_int_max=12.0)
    with pytest.raises(ValueError, match="q-calibration|too few"):
        fds.reduce_intensity(q, Iq, cfg)


def test_pattern_to_rdf_center_beam_radius():
    center = (64, 64)
    img = ring_pattern((128, 128), center=center,
                       rings=((30, 6, 1.0), (55, 6, 0.5)))
    cfg = fds.RDFConfig(q_int_min=0.3, q_int_max=2.0, r_max=8.0)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = fds.pattern_to_rdf(img, 0.03, cfg, center=center,
                                 center_beam_radius=8)
    assert np.isfinite(res.N)
    assert res.Gr.shape == res.r.shape


def test_scattering_terms_positive():
    q = np.linspace(0.1, 10, 50)
    f_sq, f_avg_sq = fds.scattering_terms(q, {"Si": 1, "O": 2})
    assert np.all(f_sq > 0) and np.all(f_avg_sq > 0)


def test_bundled_kirkland_factors_sane():
    # bundled table must load (no crude-fallback warning) and give physical f(0)
    from fourdstem.analysis.rdf import _kirkland_params, _f_kirkland
    tbl = _kirkland_params()
    for s in ("H", "O", "Si", "Au"):
        assert s in tbl
    f0 = {s: float(_f_kirkland(np.array([0.0]), tbl[s])[0])
          for s in ("H", "O", "Si", "Au")}
    # electron scattering factor at q=0 grows with Z
    assert f0["H"] < f0["O"] < f0["Si"] < f0["Au"]
    # scattering_terms must not fall back to the crude form for Si/O
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fds.scattering_terms(np.linspace(0.1, 1.5, 20), {"Si": 1, "O": 2})


def test_reduction_phi_bounded_with_real_factors():
    # regression: with correct Kirkland factors, phi must not ramp away at high q
    q = np.linspace(0.05, 1.6, 400)
    f_sq, f_avg_sq = fds.scattering_terms(q, {"Si": 1, "O": 2})
    struct = 0.35 * np.sin(2 * np.pi * q / 0.24) * np.exp(-0.8 * q)
    Iq = 120.0 * (f_sq + f_avg_sq * struct)
    cfg = fds.RDFConfig(composition={"Si": 1, "O": 2}, q_int_min=0.15,
                        q_int_max=1.5, r_min=1.1, r_max=8.0)
    qf, phi, r, Gr, diag = fds.reduce_intensity(q, Iq, cfg)
    assert diag["N"] == pytest.approx(120.0, rel=0.1)
    assert np.abs(phi).max() < 1.0          # bounded, not ramping to +2.4


# -- decomposition (needs sklearn) -----------------------------------------
def test_radial_profiles_shape_and_ring():
    from tests.synthetic import ring_pattern
    cube = np.stack([[ring_pattern((48, 48), rings=((r, 3, 1.0),), central_beam=0)
                      for r in (8, 16)]], 0)  # (1, 2, 48, 48)
    dc = fds.from_array(cube, q_per_px=1.0)
    prof, rc = fds.radial_profiles(dc, center=(24, 24), n_bins=24)
    assert prof.shape == (2, 24)
    # position 0 peaks near r=8, position 1 near r=16
    assert abs(rc[np.argmax(prof[0])] - 8) < 3
    assert abs(rc[np.argmax(prof[1])] - 16) < 3


def test_cluster_cube_two_phases():
    pytest.importorskip("sklearn")
    from tests.synthetic import ring_pattern
    a = ring_pattern((48, 48), rings=((8, 3, 1.0),), central_beam=0)
    b = ring_pattern((48, 48), rings=((16, 3, 1.0),), central_beam=0)
    Ry, Rx = 6, 8
    cube = np.empty((Ry, Rx, 48, 48))
    for iy in range(Ry):
        for ix in range(Rx):
            cube[iy, ix] = (1 + 3 * ix / Rx) * (a if ix < Rx // 2 else b)  # + brightness ramp
    dc = fds.from_array(cube, q_per_px=1.0)
    labels, patterns, km = fds.cluster_cube(dc, n_clusters=2, center=(24, 24),
                                            feature="radial", normalize="sum")
    assert labels.shape == (Ry, Rx)
    assert patterns.shape == (2, 48, 48)
    # each column should be a single cluster (structure, not brightness ramp)
    left = labels[:, :Rx // 2].ravel()
    right = labels[:, Rx // 2:].ravel()
    assert len(set(left)) == 1 and len(set(right)) == 1 and left[0] != right[0]


def test_cluster_cube_structural_catches_thin_interface():
    pytest.importorskip("sklearn")
    pytest.importorskip("scipy")
    # A THIN vertical interface (one column) with a different ring sits in a bulk
    # field, plus a strong TOP->BOTTOM brightness gradient. feature="radial" would
    # split by the row gradient (dominant variance); feature="structural" with
    # detrend must instead isolate the interface column.
    from tests.synthetic import ring_pattern
    bulk = ring_pattern((48, 48), rings=((10, 3, 1.0),), central_beam=0)
    iface = ring_pattern((48, 48), rings=((8, 3, 1.0),), central_beam=0)
    Ry, Rx = 10, 12
    x_if = 7
    cube = np.empty((Ry, Rx, 48, 48))
    for iy in range(Ry):
        for ix in range(Rx):
            bright = 1.0 + 4.0 * iy / Ry           # vertical brightness gradient
            cube[iy, ix] = bright * (iface if ix == x_if else bulk)
    dc = fds.from_array(cube, q_per_px=1.0)
    labels, patterns, km = fds.cluster_cube(
        dc, n_clusters=2, center=(24, 24), feature="structural",
        rings=[(6, 12), (7, 11)], detrend=True, detrend_sigma=3.0)
    assert labels.shape == (Ry, Rx)
    # the interface column must be a single, distinct label from the bulk
    if_labels = labels[:, x_if]
    bulk_labels = np.delete(labels, x_if, axis=1)
    assert len(set(if_labels.tolist())) == 1
    assert if_labels[0] != np.bincount(bulk_labels.ravel()).argmax()


def test_localize_interface_vertical_band():
    from tests.synthetic import ring_pattern
    bulk = ring_pattern((48, 48), rings=((10, 3, 1.0),), central_beam=0)
    iface = ring_pattern((48, 48), rings=((8, 3, 1.0),), central_beam=0)
    Sy, Sx = 12, 20
    x_if = 13
    cube = np.empty((Sy, Sx, 48, 48))
    for iy in range(Sy):
        for ix in range(Sx):
            bright = 1.0 + 3.0 * iy / Sy       # vertical brightness gradient (nuisance)
            # a 3-column-wide interface band
            w = 1.0 if abs(ix - x_if) <= 1 else 0.0
            cube[iy, ix] = bright * (iface if w else bulk)
    dc = fds.from_array(cube, q_per_px=1.0)
    info = fds.localize_interface(dc, center=(24, 24), feature="structural",
                                  rings=[(6, 12)], band_sigma=2.0)
    assert abs(info["x_if"] - x_if) <= 1.5        # found the band, not the gradient
    assert info["width"] < 8                       # a thin line, not a broad hump
    assert info["interface_mask"].sum() > 0
    assert info["bulk_mask"].sum() > 0
    # interface mask must be concentrated at x_if, bulk far away
    im_cols = np.where(info["interface_mask"].any(0))[0]
    assert im_cols.min() >= x_if - 4 and im_cols.max() <= x_if + 4
    # masks are disjoint
    assert not np.any(info["interface_mask"] & info["bulk_mask"])


def test_localize_interface_per_row_rejects_edges():
    from tests.synthetic import ring_pattern
    bulk = ring_pattern((48, 48), rings=((10, 3, 1.0),), central_beam=0)
    iface = ring_pattern((48, 48), rings=((8, 3, 1.0),), central_beam=0)
    Sy, Sx = 16, 40
    x_if = 22
    cube = np.empty((Sy, Sx, 48, 48))
    for iy in range(Sy):
        for ix in range(Sx):
            bright = 1.0 + 2.0 * iy / Sy               # horizontal striping
            w = 1.0 if abs(ix - x_if) <= 1 else 0.0
            cube[iy, ix] = bright * (iface if w else bulk)
            if ix in (0, Sx - 1):                      # strong edge artifact
                cube[iy, ix] = 5.0 * bulk
    dc = fds.from_array(cube, q_per_px=1.0)
    info = fds.localize_interface(dc, center=(24, 24), feature="structural",
                                  rings=[(6, 12)], per_row=True)
    assert abs(info["x_if"] - x_if) <= 2               # not the edge spike
    im = info["interface_mask"]
    cols = np.where(im.any(0))[0]
    assert cols.min() >= x_if - 4 and cols.max() <= x_if + 4   # band near the line
    assert 0 not in cols and (Sx - 1) not in cols      # edges excluded
    assert info["interface_area"] == int(im.sum())
    assert not np.any(im & info["bulk_mask"])


def test_localize_interface_map_and_anchor():
    # re-track from a precomputed 2-D map at an external anchor (drift tracking),
    # no DataCube needed
    from tests.synthetic import ring_pattern
    bulk = ring_pattern((48, 48), rings=((10, 3, 1.0),), central_beam=0)
    iface = ring_pattern((48, 48), rings=((8, 3, 1.0),), central_beam=0)
    Sy, Sx = 12, 40
    x_if = 22
    cube = np.empty((Sy, Sx, 48, 48))
    for iy in range(Sy):
        for ix in range(Sx):
            cube[iy, ix] = (1 + 2 * iy / Sy) * (iface if abs(ix - x_if) <= 1 else bulk)
    dc = fds.from_array(cube, q_per_px=1.0)
    smap = fds.localize_interface(dc, center=(24, 24), feature="structural",
                                  rings=[(6, 12)], per_row=True)["s"]
    # feed the stored map back with an anchor; must localize at the anchor
    info = fds.localize_interface(smap, anchor=x_if, per_row=True, line_sign="bright")
    assert abs(info["x_if"] - x_if) <= 1
    # a deliberately wrong anchor confines the search elsewhere
    info2 = fds.localize_interface(smap, anchor=6, per_row=True, line_sign="bright")
    assert info2["x_if"] < x_if - 4


def test_localize_interface_absent_when_homogeneous():
    # a homogeneous scan (no interface) must report present=False, area 0
    rng = np.random.default_rng(0)
    from tests.synthetic import ring_pattern
    bulk = ring_pattern((48, 48), rings=((10, 3, 1.0),), central_beam=0)
    Sy, Sx = 16, 40
    cube = np.empty((Sy, Sx, 48, 48))
    for iy in range(Sy):
        for ix in range(Sx):
            cube[iy, ix] = (1.0 + 2.0 * iy / Sy) * bulk   # only a brightness gradient
    cube += 0.01 * rng.standard_normal(cube.shape)
    dc = fds.from_array(cube, q_per_px=1.0)
    info = fds.localize_interface(dc, center=(24, 24), feature="structural",
                                  rings=[(6, 12)], per_row=True, min_snr=4.0)
    assert info["present"] is False
    assert info["interface_area"] == 0


def test_rdf_quality_report():
    q = np.linspace(0.05, 1.6, 400)
    f_sq, f_avg_sq = fds.scattering_terms(q, {"Si": 1, "O": 2})
    struct = 0.35 * np.sin(2 * np.pi * q / 0.24) * np.exp(-0.8 * q)
    Iq = 120.0 * (f_sq + f_avg_sq * struct)
    cfg = fds.RDFConfig(composition={"Si": 1, "O": 2}, q_int_min=0.15,
                        q_int_max=1.5, r_min=1.1, r_max=8.0)
    qf, phi, r, Gr, diag = fds.reduce_intensity(q, Iq, cfg)
    res = fds.RDFResult(q=q, Iq=Iq, q_reduced=qf, phi=phi, r=r, Gr=Gr,
                        N=diag["N"], diagnostics=diag)
    rep = fds.rdf_quality(res, expected_first_peak=1.0)
    assert "verdict" in rep and set(rep["flags"]) == {
        "first_peak_ok", "phi_not_ramping", "low_r_clean"}
    assert rep["flags"]["phi_not_ramping"]     # phi is bounded here


def test_nmf_normalize_separates_by_structure():
    pytest.importorskip("sklearn")
    # two structures (ring at r=8 vs r=13) but with a strong per-position
    # BRIGHTNESS gradient. Without normalize, NMF splits by brightness; with
    # normalize="sum" it must split by structure (the two rings).
    from tests.synthetic import ring_pattern
    a = ring_pattern((48, 48), rings=((8, 3, 1.0),), central_beam=0)
    b = ring_pattern((48, 48), rings=((13, 3, 1.0),), central_beam=0)
    Ry, Rx = 6, 8
    cube = np.empty((Ry, Rx, 48, 48))
    for iy in range(Ry):
        for ix in range(Rx):
            bright = 1.0 + 5.0 * ix / Rx           # brightness ramp across x
            cube[iy, ix] = bright * (a if ix < Rx // 2 else b)
    dc = fds.from_array(cube, q_per_px=1.0)
    res = fds.nmf_decompose(dc, n_components=2, normalize="sum")
    # each component's loading map should be dominated by one half (structure),
    # not follow the brightness ramp. Column means of the two loadings should
    # anti-correlate (one high on left half, other on right half).
    m0 = res.loadings[0].mean(0)
    m1 = res.loadings[1].mean(0)
    assert np.corrcoef(m0, m1)[0, 1] < 0     # complementary structural maps


def test_nmf_separates_amorphous_and_crystalline():
    pytest.importorskip("sklearn")
    cube = fds.from_array(scan_cube((6, 6), (64, 64)), q_per_px=0.02)
    res = fds.nmf_decompose(cube, n_components=2, max_iter=200)
    assert res.components.shape == (2, 64, 64)
    assert res.loadings.shape == (2, 6, 6)
    # loadings should be non-negative
    assert (res.loadings >= -1e-6).all()


def test_nmf_warns_on_signed_data():
    pytest.importorskip("sklearn")
    # signed profiles (like G(r)) should trigger the NMF non-negativity warning
    r = np.linspace(0, 10, 100)
    X = np.vstack([np.sin(r + 0.1 * i) for i in range(6)])  # ranges [-1, 1]
    with pytest.warns(UserWarning, match="non-negative|pca"):
        fds.decompose_profiles(X, n_components=2, method="nmf")
    # PCA on the same signed data must NOT warn
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = fds.decompose_profiles(X, n_components=2, method="pca")
    assert res.explained_variance_ratio is not None


def test_pca_reconstruct_roundtrip():
    pytest.importorskip("sklearn")
    cube = fds.from_array(scan_cube((5, 5), (48, 48)))
    res = fds.pca_decompose(cube, n_components=10)
    recon = fds.reconstruct(res)
    assert recon.shape == cube.data.shape
    # 10 comps on 25 patterns should reconstruct well
    err = np.linalg.norm(recon - cube.data) / np.linalg.norm(cube.data)
    assert err < 0.3


# -- in-situ series ---------------------------------------------------------
def test_series_from_cube_and_tracking():
    # build a stack whose ring slowly grows -> peak should move
    frames = []
    for i, radius in enumerate(np.linspace(30, 40, 6)):
        frames.append(ring_pattern((96, 96), rings=((radius, 5, 1.0),),
                                    central_beam=0.0))
    stack = np.stack(frames, 0)
    series = fds.Series.from_cube(stack, coords=np.arange(6) * 100.0,
                                  q_per_px=1.0)
    assert len(series) == 6
    profiles = [fds.azimuthal_integrate(f.pattern, (48, 48), 1.0)
                for f in series]
    track = fds.track_peak(profiles, series.coordinates(), (25, 45))
    # position should increase monotonically-ish with coordinate
    assert track["position"][-1] > track["position"][0]


def test_coordinate_from_name():
    assert fds.coordinate_from_name("SiOx_450K_scan") == 450
    assert fds.coordinate_from_name("run_300_a") == 300
    assert fds.coordinate_from_name("600K") == 600
    # leading-zero temperature filenames (0025K.dm4 style)
    assert fds.coordinate_from_name("0025K") == 25
    assert fds.coordinate_from_name("0100K") == 100
    assert fds.coordinate_from_name("1100K") == 1100
    assert np.isnan(fds.coordinate_from_name("nothing_here"))


def test_from_directory_flat_files(tmp_path):
    # flat layout: coordinate-named files directly in the folder
    from fourdstem.io.writers import save_datacube_npz
    for T in (25, 100, 300):
        cube = fds.from_array(ring_pattern((48, 48)), q_per_px=0.02)
        save_datacube_npz(str(tmp_path / f"{T:04d}K.npz"), cube)
    series = fds.Series.from_directory(str(tmp_path), pattern="*.npz")
    assert len(series) == 3
    assert list(series.coordinates()) == [25.0, 100.0, 300.0]


def test_from_directory_subfolders(tmp_path):
    from fourdstem.io.writers import save_datacube_npz
    for T in (300, 500):
        d = tmp_path / f"{T}K"
        d.mkdir()
        cube = fds.from_array(ring_pattern((48, 48)), q_per_px=0.02)
        save_datacube_npz(str(d / "scan.npz"), cube)
    series = fds.Series.from_directory(str(tmp_path), pattern="*.npz")
    assert len(series) == 2
    assert list(series.coordinates()) == [300.0, 500.0]


def test_from_directory_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        fds.Series.from_directory(str(tmp_path), pattern="*.dm4")


# -- parallel / progress ----------------------------------------------------
def test_parallel_map_sequential_equals_parallel():
    def sq(x):
        return x * x
    items = list(range(10))
    seq = fds.parallel_map(sq, items, n_jobs=1, progress=False)
    assert seq == [x * x for x in items]
    par = fds.parallel_map(sq, items, n_jobs=2, progress=False)
    assert par == seq                      # order preserved


def test_parallel_map_lambda_closure():
    # loky/cloudpickle must handle a closure over a captured variable
    k = 3
    out = fds.parallel_map(lambda x: x + k, [1, 2, 3], n_jobs=2, progress=False)
    assert out == [4, 5, 6]


def test_series_map_parallel_matches_sequential():
    stack = np.stack([ring_pattern((48, 48), rings=((r, 5, 1.0),))
                      for r in (15, 20, 25, 30)], 0)
    series = fds.Series.from_cube(stack, coords=np.arange(4) * 100.0, q_per_px=1.0)
    seq = series.map(lambda f: float(f.pattern.sum()), n_jobs=1, progress=False)
    par = series.map(lambda f: float(f.pattern.sum()), n_jobs=2, progress=False)
    assert np.allclose(seq, par)


# -- preprocessing cleanup --------------------------------------------------
def test_remove_hot_pixels():
    # realistic detector: smooth signal + read noise + a few bright spikes
    rng = np.random.default_rng(0)
    img = ring_pattern((64, 64), rings=((20, 6, 1.0),), central_beam=2.0)
    img = img + 0.05 * rng.standard_normal(img.shape)
    for (y, x) in [(30, 40), (12, 50), (55, 8)]:
        img[y, x] += 5e3
    cleaned, mask = fds.remove_hot_pixels(img, threshold=8, return_mask=True)
    for (y, x) in [(30, 40), (12, 50), (55, 8)]:
        assert mask[y, x]
        assert cleaned[y, x] < 50          # replaced by local median
    assert mask.sum() < 20                 # few false positives


def test_remove_dead_pixels():
    img = ring_pattern((48, 48), rings=((15, 4, 1.0),), central_beam=5) + 1.0
    img[10, 10] = 0.0
    cleaned, mask = fds.remove_dead_pixels(img, return_mask=True)
    assert mask[10, 10]
    assert cleaned[10, 10] > 0


# -- Gaussian peak fit ------------------------------------------------------
def test_fit_gaussian_peak_recovers_center():
    x = np.linspace(1.0, 2.5, 300)
    y = 3.0 * np.exp(-((x - 1.62) ** 2) / (2 * 0.05 ** 2)) + 0.2
    fit = fds.fit_gaussian_peak(x, y, 1.5, 1.7)
    assert fit["success"]
    assert fit["center"] == pytest.approx(1.62, abs=0.01)
    assert fit["sigma"] == pytest.approx(0.05, abs=0.01)


# -- profile decomposition (needs sklearn) ---------------------------------
def test_decompose_profiles_two_endmembers():
    pytest.importorskip("sklearn")
    r = np.linspace(0, 10, 200)
    a = np.exp(-((r - 1.6) ** 2) / (2 * 0.1 ** 2))       # end-member A
    b = np.exp(-((r - 2.6) ** 2) / (2 * 0.1 ** 2))       # end-member B
    fracs_true = np.linspace(0, 1, 8)
    X = np.vstack([(1 - t) * a + t * b for t in fracs_true])
    res = fds.decompose_profiles(X, n_components=2, x=r)
    assert res.components.shape == (2, 200)
    assert res.weights.shape == (8, 2)
    assert res.fractions.shape == (8, 2)
    # each sample's fractions should sum to 1
    assert np.allclose(res.fractions.sum(axis=1), 1.0, atol=1e-6)
    # one component's fraction should trend monotonically with the true mix
    trend = res.fractions[:, 0]
    assert abs(np.corrcoef(fracs_true, trend)[0, 1]) > 0.95


# -- NBED per-pattern helpers (median, reduce_profiles) --------------------
def test_median_pattern_rejects_outliers():
    from tests.synthetic import ring_pattern
    base = ring_pattern((32, 32), rings=((10, 3, 1.0),), central_beam=2.0)
    Ry, Rx = 5, 6
    cube = np.stack([[base.copy() for _ in range(Rx)] for _ in range(Ry)])
    # inject a few hot frames (huge values) — median must ignore them, mean would not
    cube[0, 0] += 1e5
    cube[1, 2] += 1e5
    dc = fds.from_array(cube.astype(float), q_per_px=0.03)
    med = fds.median_pattern(dc)
    assert med.shape == (32, 32)
    assert np.allclose(med, base, atol=1e-6)              # outliers rejected
    assert fds.to_pattern(dc).max() > 100                 # mean is polluted


def test_reduce_profiles_stack_matches_single():
    q = np.linspace(0.05, 1.5, 300)
    f_sq, f_avg_sq = fds.scattering_terms(q, {"Si": 1, "O": 2})
    cfg = fds.RDFConfig(composition={"Si": 1, "O": 2}, q_int_min=0.15,
                        q_int_max=1.4, r_min=1.1, r_max=8.0)
    # two distinct structure signals -> a small stack
    prof = []
    for period in (0.24, 0.30):
        s = 0.35 * np.sin(2 * np.pi * q / period) * np.exp(-0.8 * q)
        prof.append(120.0 * (f_sq + f_avg_sq * s))
    prof = np.array(prof)
    out = fds.reduce_profiles(prof, q, cfg, n_jobs=1)
    assert out["phi"].shape[0] == 2 and out["Gr"].shape[0] == 2
    assert out["ok"].all()
    # stack row matches an independent single reduction
    qf, phi1, r, Gr1, diag = fds.reduce_intensity(q, prof[0], cfg)
    assert np.allclose(out["phi"][0], phi1, atol=1e-6)
    assert np.allclose(out["Gr"][0], Gr1, atol=1e-6)


# -- dm4 reader: pick the real cube, not a 2D thumbnail (STEM SI files) -----
def test_read_ncempy_prefers_highest_dim_dataset(monkeypatch):
    # a STEM SI .dm4 holds survey (2D) + scan (2D) + CBED (4D); dmReader may
    # return the 2D survey. _read_ncempy must search datasets and pick the 4D.
    import sys, types
    thumb = {"data": np.zeros((8, 8)), "pixelSize": [1.0, 1.0],
             "pixelUnit": ["nm", "nm"]}
    cube4d = {"data": np.zeros((3, 4, 8, 8)), "pixelSize": [1, 1, 0.1, 0.1],
              "pixelUnit": ["nm", "nm", "1/nm", "1/nm"]}
    fake_dm = types.ModuleType("ncempy.io.dm")
    fake_dm.dmReader = lambda path: thumb

    class _FakeFileDM:
        def __init__(self, p): self.numObjects = 2
        def parseHeader(self): pass
        def getDataset(self, i):
            if i == 0: return thumb
            if i == 1: return cube4d
            raise IndexError(i)
    fake_dm.fileDM = _FakeFileDM
    fake_io = types.ModuleType("ncempy.io"); fake_io.dm = fake_dm
    fake = types.ModuleType("ncempy"); fake.io = fake_io
    monkeypatch.setitem(sys.modules, "ncempy", fake)
    monkeypatch.setitem(sys.modules, "ncempy.io", fake_io)
    monkeypatch.setitem(sys.modules, "ncempy.io.dm", fake_dm)

    from fourdstem.io.readers import _read_ncempy
    data, scale, unit, meta = _read_ncempy("fake.dm4")
    assert data.ndim == 4 and data.shape == (3, 4, 8, 8)   # got the CBED cube


# -- vacuum / reference subtraction ----------------------------------------
def test_subtract_reference_zeros_empty_region():
    from tests.synthetic import ring_pattern
    mat = ring_pattern((32, 32), rings=((10, 3, 1.0),), central_beam=3.0)
    vac = ring_pattern((32, 32), rings=(), central_beam=3.0)   # beam only, no ring
    Ry, Rx = 6, 5
    cube = np.empty((Ry, Rx, 32, 32))
    cube[:] = mat
    cube[-2:, :] = vac                                          # bottom 2 rows = empty
    dc = fds.from_array(cube, q_per_px=0.03)
    empty = np.zeros((Ry, Rx), bool); empty[-2:, :] = True
    ref = fds.average_pattern(dc, empty)
    sub = fds.subtract_reference(dc, ref)
    assert sub.calibration.q_per_px == pytest.approx(0.03)
    assert sub.metadata.get("reference_subtracted") is True
    # empty rows collapse to ~0; material rows keep the ring (nonzero)
    flat = sub._flat_patterns().reshape(Ry, Rx, 32, 32)
    assert np.abs(flat[-2:]).mean() < 1e-4
    assert np.abs(flat[:-2]).mean() > 0.05
    # ndarray input path + clip
    arr_sub = fds.subtract_reference(cube, ref, clip_negative=True)
    assert arr_sub.min() >= 0.0 and arr_sub.shape == cube.shape
