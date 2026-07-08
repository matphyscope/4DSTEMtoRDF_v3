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
    assert np.isnan(fds.coordinate_from_name("nothing_here"))
