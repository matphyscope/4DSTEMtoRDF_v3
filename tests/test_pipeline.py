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


def test_scattering_terms_positive():
    q = np.linspace(0.1, 10, 50)
    f_sq, f_avg_sq = fds.scattering_terms(q, {"Si": 1, "O": 2})
    assert np.all(f_sq > 0) and np.all(f_avg_sq > 0)


# -- decomposition (needs sklearn) -----------------------------------------
def test_nmf_separates_amorphous_and_crystalline():
    pytest.importorskip("sklearn")
    cube = fds.from_array(scan_cube((6, 6), (64, 64)), q_per_px=0.02)
    res = fds.nmf_decompose(cube, n_components=2, max_iter=200)
    assert res.components.shape == (2, 64, 64)
    assert res.loadings.shape == (2, 6, 6)
    # loadings should be non-negative
    assert (res.loadings >= -1e-6).all()


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
