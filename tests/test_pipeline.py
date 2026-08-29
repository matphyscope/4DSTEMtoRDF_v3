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


def _sim_cubic_zone(phase, hmax=3):
    # [001]-zone single-crystal g-vectors (l=0) for a cubic candidate
    from fourdstem.analysis.indexing import _recip_basis, LATTICE, CENTERING_RULE
    B = _recip_basis(LATTICE[phase]); rule = CENTERING_RULE[LATTICE[phase]["centering"]]
    gs = []
    for h in range(-hmax, hmax + 1):
        for k in range(-hmax, hmax + 1):
            if (h, k) == (0, 0) or not rule(h, k, 0):
                continue
            g = h * B[0] + k * B[1]
            if np.hypot(g[0], g[1]) <= 1.15:
                gs.append([g[0], g[1]])
    return np.asarray(gs, float)


def test_index_pattern_confirms_and_rejects_denser_lattice():
    # a LiF [001] single-crystal pattern must index as LiF, and the near-coincident
    # denser Li2S lattice (a ~ sqrt(2)*a_LiF) must lose on completeness, not win on
    # coverage. This is the "confirmed by indexing" tier, not a |q| match.
    gs = _sim_cubic_zone("LiF")
    best, results = fds.index_pattern(gs, candidates=["LiF", "Li2O", "Li2S"], min_spots=3)
    assert best["phase"] == "LiF" and best["indexed"]
    assert best["completeness"] > 0.9 and abs(best["zone"][2]) > 0    # [00l] zone
    by = {r["phase"]: r for r in results}
    assert by["Li2S"]["completeness"] < by["LiF"]["completeness"]     # denser lattice penalized


def test_index_pattern_needs_enough_spots():
    # two spots cannot fix a zone -> no indexing
    gs = _sim_cubic_zone("LiF")[:2]
    best, results = fds.index_pattern(gs, candidates=["LiF"], min_spots=3)
    assert best is None or not best.get("indexed", False)


def test_index_grains_end_to_end():
    # a cube with one crystalline grain (LiF [001] spots) in a corner, amorphous
    # elsewhere: crystallinity map + grain labels + indexing should recover a grain
    # and index it as LiF.
    from fourdstem.analysis.indexing import _recip_basis, LATTICE
    B = _recip_basis(LATTICE["LiF"])
    H = W = 96
    cx, cy = W / 2, H / 2
    q_per_px = 0.02
    yy, xx = np.mgrid[0:H, 0:W]
    rr = np.hypot(xx - cx, yy - cy)
    beam = 20 * np.exp(-rr ** 2 / (2 * 2.0 ** 2))
    halo = 3 * np.exp(-(rr - 22) ** 2 / (2 * 4.0 ** 2))
    # LiF [001] spots as a detector pattern
    spot_img = np.zeros((H, W))
    for h in range(-3, 4):
        for k in range(-3, 4):
            if (h, k) == (0, 0) or not (h % 2 == k % 2 == 0):
                continue
            g = h * B[0] + k * B[1]
            px, py = cx + g[0] / q_per_px, cy + g[1] / q_per_px
            if 0 <= px < W and 0 <= py < H:
                spot_img += 30 * np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * 1.4 ** 2))
    Sy, Sx = 12, 12
    rng = np.random.default_rng(0)
    cube = np.empty((Sy, Sx, H, W), np.float32)
    for iy in range(Sy):
        for ix in range(Sx):
            base = beam + halo
            if iy < 4 and ix < 4:                      # crystalline grain corner
                base = beam + spot_img
            cube[iy, ix] = np.clip(base + 0.2 * rng.standard_normal((H, W)), 0, None)
    dc = fds.from_array(cube, q_per_px=q_per_px)
    grains, labels = fds.index_grains(dc, center=(cx, cy),
                                      candidates=["LiF", "Li2O", "Li2S"],
                                      threshold_pctl=85, min_size=3,
                                      spot_kwargs=dict(n_mad=4.0),
                                      index_kwargs=dict(min_spots=3))
    assert labels.max() >= 1                            # found a grain
    indexed = [g for g in grains if g["best"] and g["best"].get("indexed")]
    assert any(g["best"]["phase"] == "LiF" for g in indexed)


def test_vacuum_referenced_halo_detector():
    # amorphous halo at r=15 (q=0.3) in material, beam-only vacuum, with a
    # brightness (thickness) gradient. A vacuum-referenced ring detector must
    # separate material from vacuum cleanly at a sigma cutoff, and the peak-above-
    # flank makes the DETECTION insensitive to the brightness ramp.
    from fourdstem.analysis.detectors import detector_map, strict_vacuum_mask, amorphous_halo_peaks
    from fourdstem.analysis.classify import radial_stack
    H = W = 96
    cx, cy = W / 2, H / 2
    q_per_px = 0.02
    yy, xx = np.mgrid[0:H, 0:W]
    rr = np.hypot(xx - cx, yy - cy)
    beam = 30 * np.exp(-rr ** 2 / (2 * 2.5 ** 2))
    halo = np.exp(-(rr - 15) ** 2 / (2 * 4.0 ** 2))
    Sy, Sx = 20, 20
    rng = np.random.default_rng(0)
    cube = np.empty((Sy, Sx, H, W), np.float32)
    truth = np.zeros((Sy, Sx), bool)
    for iy in range(Sy):
        for ix in range(Sx):
            bright = 1 + 3 * ix / Sx
            if iy < Sy - 5:
                base = bright * (beam + 2 * halo); truth[iy, ix] = True
            else:
                base = 0.9 * bright * beam
            cube[iy, ix] = np.clip(base + 0.1 * np.sqrt(np.clip(base, 0, None)) *
                                   rng.standard_normal((H, W)), 0, None)
    dc = fds.from_array(cube, q_per_px=q_per_px)
    vac = strict_vacuum_mask(dc, center=(cx, cy), pctl=15)
    assert 0.1 <= vac.mean() <= 0.2
    q, prof = radial_stack(dc, (cx, cy), q_per_px, q_max=1.2, nbin=200)
    peaks = amorphous_halo_peaks(q, prof[truth.ravel()].mean(0))
    assert any(abs(p - 0.30) < 0.05 for p in peaks)          # halo radius found
    sig, _ = detector_map(dc, (cx, cy), q_per_px, 0.30, dq=0.04,
                          vacuum_mask=vac, _stack=(q, prof))
    assert (sig[truth] > 3).mean() > 0.9                     # material above 3 sigma
    assert (sig[~truth] > 3).mean() < 0.1                    # vacuum below cutoff


def test_structural_halo_rejects_smooth_background():
    # three regions: a real amorphous halo BUMP, a smooth (featureless) background
    # with intensity at the halo radius, and vacuum. A plain annular detector is
    # fooled by the smooth background; the structural halo map (bump above a
    # linear flank baseline) must flag only the real bump.
    from fourdstem.analysis.detectors import structural_halo_map, detector_map, strict_vacuum_mask
    from fourdstem.analysis.classify import radial_stack
    H = W = 100
    cx, cy = W / 2, H / 2
    q_per_px = 0.02
    yy, xx = np.mgrid[0:H, 0:W]
    rr = np.hypot(xx - cx, yy - cy)
    beam = 30 * np.exp(-rr ** 2 / (2 * 2.5 ** 2))
    halo = np.exp(-(rr - 13) ** 2 / (2 * 6.0 ** 2))
    smooth = 1.0 / (1.0 + (rr / 8.0) ** 2)
    Sy, Sx = 24, 24
    rng = np.random.default_rng(0)
    cube = np.empty((Sy, Sx, H, W), np.float32)
    lab = np.zeros((Sy, Sx), int)
    for iy in range(Sy):
        for ix in range(Sx):
            if iy < 8:
                base = beam + 3 * halo; lab[iy, ix] = 1
            elif iy < 16:
                base = beam + 20 * smooth; lab[iy, ix] = 2
            else:
                base = 0.9 * beam; lab[iy, ix] = 0
            cube[iy, ix] = np.clip(base + 0.1 * np.sqrt(np.clip(base, 0, None)) *
                                   rng.standard_normal((H, W)), 0, None)
    dc = fds.from_array(cube, q_per_px=q_per_px)
    vac = strict_vacuum_mask(dc, center=(cx, cy), pctl=15)
    q, prof = radial_stack(dc, (cx, cy), q_per_px, q_max=1.2, nbin=200)
    plain, _ = detector_map(dc, (cx, cy), q_per_px, 0.26, dq=0.06, flank=None,
                            vacuum_mask=vac, _stack=(q, prof))
    struct, _ = structural_halo_map(dc, (cx, cy), q_per_px, 0.26, dq=0.06,
                                    flank_gap=0.10, flank_w=0.05, vacuum_mask=vac, _stack=(q, prof))
    assert plain[lab == 2].mean() > plain[lab == 1].mean() * 0.5   # plain fooled by smooth bg
    assert struct[lab == 1].mean() > 3                             # real halo bump flagged
    assert struct[lab == 2].mean() < 3                             # smooth bg rejected


def test_structural_halo_beam_guard_low_q_fsdp():
    # A *low-q* FSDP (bump close to the beam) over a steep convex power-law tail.
    # A wide symmetric flank_gap would place the low flank inside the beam and zero
    # the bump everywhere; the beam guard must clamp it above the beam so the real
    # bump is flagged and a featureless steep background (no bump) is rejected.
    from fourdstem.analysis.detectors import structural_halo_map
    rng = np.random.default_rng(0)
    Ny, Nx, det = 12, 12, 140
    cy = cx = det / 2
    yy, xx = np.mgrid[0:det, 0:det]
    r = np.hypot(yy - cy, xx - cx)
    qpp = 0.0043888
    q = r * qpp

    def bg(s):
        return s / np.clip(q, 0.02, None) ** 2.2 + 2.0

    def fsdp(a, q0=0.25, w=0.05):
        return a * np.exp(-0.5 * ((q - q0) / w) ** 2)

    cube = np.zeros((Ny, Nx, det, det), np.float32)
    has = np.zeros((Ny, Nx), bool)
    for iy in range(Ny):
        for ix in range(Nx):
            if iy < 2:
                base = bg(0.02) * 0.05                    # vacuum
            elif ix < 6:
                base = bg(1.0) + fsdp(30.0); has[iy, ix] = True   # real halo
            else:
                base = bg(1.6)                            # steep smooth bg, NO halo
            cube[iy, ix] = rng.poisson(np.clip(base, 0, None))
    dc = fds.from_array(cube, q_per_px=qpp)
    vac = np.zeros((Ny, Nx), bool); vac[:2] = True
    mat = ~vac
    # a deliberately wide gap whose low flank (0.25-0.16-0.05=0.04) sits in the beam
    sig, raw = structural_halo_map(dc, (cx, cy), qpp, 0.25, dq=0.06,
                                   flank_gap=0.16, flank_w=0.05, vacuum_mask=vac)
    assert raw[has].mean() > 1.0                          # real bump survives the guard
    assert raw[mat & ~has].mean() < 0.3 * raw[has].mean() # steep smooth bg -> ~0 bump
    assert np.mean(sig[has] > 3) > 0.8
    assert np.mean(sig[mat & ~has] > 3) < 0.1


def test_amorphous_halo_peaks_width_filter_rejects_sharp_ring():
    # A broad halo at q=0.22 and a SHARP crystalline ring at q=0.33 both make a
    # maximum in the profile. Without width filtering both are returned; with
    # min_width_q the sharp ring is dropped (halos are broad) and only the halo
    # survives, so the ring stays available to the ring stage. Also: no peak is
    # pinned to the q_lo boundary (beam-edge artifact).
    from fourdstem.analysis.detectors import amorphous_halo_peaks
    q = np.linspace(0.0, 1.15, 230)
    def bg(s):
        return s * (0.03 / np.clip(q, 0.01, None)) ** 2.4 + 0.02 * s
    def gg(a, q0, w):
        return a * np.exp(-0.5 * ((q - q0) / w) ** 2)
    prof = bg(280) + gg(9.0, 0.22, 0.05) + gg(6.0, 0.33, 0.012)
    both = amorphous_halo_peaks(q, prof, q_lo=0.18, q_hi=1.0, smooth=3, min_width_q=0.0)
    broad = amorphous_halo_peaks(q, prof, q_lo=0.18, q_hi=1.0, smooth=3, min_width_q=0.045)
    assert any(abs(p - 0.22) < 0.03 for p in both)
    assert any(abs(p - 0.33) < 0.03 for p in both)      # ring seen without filter
    assert any(abs(p - 0.22) < 0.03 for p in broad)     # halo kept
    assert not any(abs(p - 0.33) < 0.03 for p in broad)  # sharp ring rejected
    assert not any(p <= 0.18 + 0.03 for p in broad)      # no boundary artifact


def test_halo_bump_contrast_is_thickness_independent():
    # halo_bump_maps.contrast (bump / local background) must flag a real halo and
    # reject thick-but-featureless material, where the raw bump alone (scaling with
    # thickness) cannot. Convex power-law tail + localized FSDP in half the material;
    # the other half is thicker with no halo; vacuum near-silent.
    from fourdstem.analysis.detectors import halo_bump_maps
    rng = np.random.default_rng(2)
    Ny, Nx, det = 16, 16, 150
    cy = cx = det / 2
    yy, xx = np.mgrid[0:det, 0:det]
    r = np.hypot(yy - cy, xx - cx)
    qpp = 0.0043888
    q = r * qpp

    def bg(s):
        return s * (0.03 / np.clip(q, 0.02, None)) ** 2.4 + 0.02 * s

    def halo(a, q0=0.25, w=0.05):
        return a * np.exp(-0.5 * ((q - q0) / w) ** 2)

    cube = np.zeros((Ny, Nx, det, det), np.float32)
    has = np.zeros((Ny, Nx), bool)
    for iy in range(Ny):
        for ix in range(Nx):
            if iy < 3:
                base = bg(300) * 0.02
            elif ix < 8:
                base = bg(300) + halo(9.0); has[iy, ix] = True
            else:
                base = bg(360)                       # thicker, NO halo
            cube[iy, ix] = rng.poisson(np.clip(base, 0, None))
    dc = fds.from_array(cube, q_per_px=qpp)
    vac = np.zeros((Ny, Nx), bool); vac[:3] = True
    mat = ~vac
    out = halo_bump_maps(dc, (cx, cy), qpp, [0.25], dq=0.06, beam_cut=0.14,
                         q_max=1.0, deg=2, vacuum_mask=vac)
    csig = out[0.25]["csig"]
    # contrast is higher for the real halo than for thick-flat material...
    assert out[0.25]["contrast"][has].mean() > 1.8 * out[0.25]["contrast"][mat & ~has].mean()
    # ...and at >4 sigma the halo fires while thick-flat does not
    assert np.mean(csig[has] > 4) > 0.8
    assert np.mean(csig[mat & ~has] > 4) < 0.2


def test_ring_phase_evidence_finds_rings_not_amorphous():
    # a polycrystalline LiF ring patch in an otherwise amorphous film. Referenced
    # to the amorphous background, LiF ring evidence must be high in the patch and
    # ~0 in the amorphous region (no false ring detection on the broad halo), and
    # LiF must beat Li2O/Li2S (which only share a couple of rings).
    H = W = 120
    cx, cy = W / 2, H / 2
    q_per_px = 0.02
    yy, xx = np.mgrid[0:H, 0:W]
    rr = np.hypot(xx - cx, yy - cy)
    beam = 30 * np.exp(-rr ** 2 / (2 * 2.5 ** 2))
    broadhalo = np.exp(-(rr - 15) ** 2 / (2 * 12.0 ** 2))
    lif = np.zeros((H, W))
    for d, w in fds.COMPOUND_RINGS["LiF"]:
        r = (1.0 / d) / q_per_px
        if r < min(cx, cy):
            lif += 8 * w * np.exp(-(rr - r) ** 2 / (2 * 1.5 ** 2))
    Sy, Sx = 24, 24
    rng = np.random.default_rng(0)
    cube = np.empty((Sy, Sx, H, W), np.float32)
    mat = np.zeros((Sy, Sx), bool)
    for iy in range(Sy):
        for ix in range(Sx):
            base = beam + 1.5 * broadhalo
            mat[iy, ix] = iy < Sy - 4
            if not mat[iy, ix]:
                base = 0.9 * beam
            if 6 <= iy < 11 and 6 <= ix < 16:
                base = base + lif
            cube[iy, ix] = np.clip(base + 0.1 * np.sqrt(np.clip(base, 0, None)) *
                                   rng.standard_normal((H, W)), 0, None)
    dc = fds.from_array(cube, q_per_px=q_per_px)
    ev = fds.ring_phase_evidence(dc, center=(cx, cy), q_per_px=q_per_px,
                                 candidates=["LiF", "Li2O", "Li2S"], ref_mask=mat,
                                 ring_sigma=4.0)
    patch = np.zeros((Sy, Sx), bool); patch[6:11, 6:16] = True
    amorph = mat & ~patch
    assert ev["LiF"]["evidence"][patch].mean() > 0.6           # LiF rings present in patch
    assert ev["LiF"]["evidence"][amorph].mean() < 0.1          # not in amorphous
    assert ev["LiF"]["evidence"][patch].mean() > ev["Li2O"]["evidence"][patch].mean()
    assert ev["LiF"]["evidence"][patch].mean() > ev["Li2S"]["evidence"][patch].mean()


def test_classify_pixels_tiers_and_examples():
    # per-pixel classification: a crystalline LiF corner should reach predicted/
    # confirmed LiF, amorphous material should be 'weak', vacuum 'none', and one
    # representative pixel is picked for each of halo/ring/spot.
    from fourdstem.analysis.indexing import _recip_basis, LATTICE
    B = _recip_basis(LATTICE["LiF"])
    H = W = 80
    cx, cy = W / 2, H / 2
    q_per_px = 0.02
    yy, xx = np.mgrid[0:H, 0:W]
    rr = np.hypot(xx - cx, yy - cy)
    beam = 20 * np.exp(-rr ** 2 / (2 * 2.0 ** 2))
    halo = 3 * np.exp(-(rr - 20) ** 2 / (2 * 4.0 ** 2))
    spot = np.zeros((H, W))
    for h in range(-3, 4):
        for k in range(-3, 4):
            if (h, k) == (0, 0) or not (h % 2 == k % 2 == 0):
                continue
            g = h * B[0] + k * B[1]
            px, py = cx + g[0] / q_per_px, cy + g[1] / q_per_px
            if 0 <= px < W and 0 <= py < H:
                spot += 30 * np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * 1.4 ** 2))
    Sy, Sx = 16, 16
    rng = np.random.default_rng(0)
    cube = np.empty((Sy, Sx, H, W), np.float32)
    mat = np.zeros((Sy, Sx), bool)
    for iy in range(Sy):
        for ix in range(Sx):
            base = beam + halo
            if 3 <= iy < 6 and 3 <= ix < 6:
                base = beam + spot
            if iy < Sy - 3:
                mat[iy, ix] = True
            cube[iy, ix] = np.clip(base + 0.2 * rng.standard_normal((H, W)), 0, None)
    dc = fds.from_array(cube, q_per_px=q_per_px)
    r = fds.classify_pixels(dc, center=(cx, cy), material=mat,
                            candidates=["LiF", "Li2O", "Li2S"],
                            spot_kwargs=dict(n_mad=4.0))
    assert r.tier.shape == (Sy, Sx)
    assert (r.tier == 0)[~mat].all()                      # vacuum -> none
    assert (r.tier >= 2).any()                            # some predicted/confirmed
    # the crystalline corner carries LiF predicted or confirmed
    corner = r.phase_idx[3:6, 3:6]
    assert (corner == r.candidates.index("LiF")).any()
    for key in ("halo", "ring", "spot"):
        iy, ix = r.examples[key]
        assert mat[iy, ix]                                # examples are on material


def test_seed_positions_and_index_seeds():
    # reuse the grain-corner cube: seeding from a location-like map must place a
    # seed in the crystalline corner, and index_seeds must index it as LiF and
    # still report weak (few-spot) seeds rather than dropping them.
    from fourdstem.analysis.indexing import _recip_basis, LATTICE
    B = _recip_basis(LATTICE["LiF"])
    H = W = 96
    cx, cy = W / 2, H / 2
    q_per_px = 0.02
    yy, xx = np.mgrid[0:H, 0:W]
    rr = np.hypot(xx - cx, yy - cy)
    beam = 20 * np.exp(-rr ** 2 / (2 * 2.0 ** 2))
    halo = 3 * np.exp(-(rr - 22) ** 2 / (2 * 4.0 ** 2))
    spot_img = np.zeros((H, W))
    for h in range(-3, 4):
        for k in range(-3, 4):
            if (h, k) == (0, 0) or not (h % 2 == k % 2 == 0):
                continue
            g = h * B[0] + k * B[1]
            px, py = cx + g[0] / q_per_px, cy + g[1] / q_per_px
            if 0 <= px < W and 0 <= py < H:
                spot_img += 30 * np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * 1.4 ** 2))
    Sy, Sx = 12, 12
    rng = np.random.default_rng(0)
    cube = np.empty((Sy, Sx, H, W), np.float32)
    locmap = np.zeros((Sy, Sx))
    for iy in range(Sy):
        for ix in range(Sx):
            base = beam + halo
            if iy < 3 and ix < 3:
                base = beam + spot_img
                locmap[iy, ix] = 1.0                    # a "location map" peak on the grain
            cube[iy, ix] = np.clip(base + 0.2 * rng.standard_normal((H, W)), 0, None)
    dc = fds.from_array(cube, q_per_px=q_per_px)
    seeds = fds.seed_positions([locmap], min_distance=1, threshold_pctl=90)
    assert any(iy < 3 and ix < 3 for (iy, ix) in seeds)   # seeded on the grain
    res = fds.index_seeds(dc, seeds, center=(cx, cy), window=1,
                          candidates=["LiF", "Li2O", "Li2S"],
                          spot_kwargs=dict(n_mad=4.0),
                          index_kwargs=dict(min_score=0.6, confirm_min_spots=4))
    assert len(res) == len(seeds)                         # every seed reported
    got = [r for r in res if r["best"] and r["best"].get("indexed")
           and r["best"]["phase"] == "LiF"]
    assert got                                            # grain seed indexed as LiF


def test_decompose_fractions_recovers_mixture():
    # profile = Li2S + LiF fingerprints on a smooth background; NNLS decomposition
    # must recover those two and zero the rest, and report a symmetric Gram.
    q = np.linspace(0.0, 1.1, 400)
    prof = 2.0 * fds.phase_ring_profile(q, "Li2S") + 1.5 * fds.phase_ring_profile(q, "LiF")
    bg = 300 * np.exp(-q / 0.25) + 20
    rng = np.random.default_rng(0)
    I = bg + 60 * prof + rng.normal(0, 3, q.size)
    r = fds.decompose_fractions(q, I)
    f = r["fractions"]
    assert f["Li2S"] > 0.4 and f["LiF"] > 0.1                 # both present phases found
    assert f["Li2O"] < 0.05 and f["Li3N"] < 0.05 and f["Li2CO3"] < 0.05   # absent ~0
    G = r["gram"]
    assert np.allclose(np.diag(G), 1.0) and np.allclose(G, G.T)   # valid correlation matrix
    assert abs(sum(f.values()) - 1.0) < 1e-6                  # fractions normalized


def test_decompose_halo_basis_prevents_fsdp_misassignment():
    # strong amorphous FSDP (d~4A) + faint crystalline LiF. Without a halo basis
    # NNLS mis-assigns the FSDP to a candidate; with it, the FSDP is absorbed and
    # the crystalline fingerprint (LiF) is recovered with a much smaller residual.
    q = np.linspace(0.0, 1.1, 400)
    fsdp = 0.6 * np.exp(-0.5 * ((q - 0.25) / 0.06) ** 2)
    lif = 0.03 * fds.phase_ring_profile(q, "LiF")
    rng = np.random.default_rng(1)
    I = 200 * np.exp(-q / 0.15) + 80 * (fsdp + lif) + rng.normal(0, 0.6, q.size)
    r_no = fds.decompose_fractions(q, I, q_lo=0.20, halo_q=None)
    r_yes = fds.decompose_fractions(q, I, q_lo=0.20, halo_q=0.25, halo_sigma=0.08)
    assert r_yes["resid_frac"] < r_no["resid_frac"]           # halo basis fits better
    assert r_yes["halo_amount"] > 0                            # FSDP absorbed by halo
    assert 0.0 < r_yes["crystallinity"] < 0.5                  # mostly amorphous
    assert r_yes["fractions"]["LiF"] > 0.2                     # crystalline LiF recovered


def test_thickness_map_vacuum_referenced_recovers_ramp():
    # left third = vacuum (no scattering), rest = thickness ramp. A per-pixel
    # detector dark offset would fake a thick vacuum; vacuum referencing must
    # (a) self-estimate that dark and (b) zero the vacuum, recovering t/lambda.
    from fourdstem.analysis.virtual_image import thickness_map
    rng = np.random.default_rng(0)
    Sy, Sx, H, W = 12, 18, 64, 64
    yy, xx = np.mgrid[0:H, 0:W]
    rr = np.hypot(xx - W / 2, yy - H / 2)
    beam = np.exp(-rr ** 2 / (2 * 1.6 ** 2))          # sharp probe, well inside disk
    halo = np.exp(-(rr - 18) ** 2 / (2 * 3.0 ** 2))
    D_true, cur = 2.0, 4000.0
    cube = np.empty((Sy, Sx, H, W), np.float32)
    tl_true = np.zeros((Sy, Sx))
    for iy in range(Sy):
        for ix in range(Sx):
            tl = 0.0 if ix < Sx // 3 else 0.2 + 2.0 * (ix - Sx // 3) / (Sx - Sx // 3)
            tl_true[iy, ix] = tl
            f = 1 - np.exp(-tl)
            pat = cur * ((1 - f) * beam / beam.sum() + f * halo / halo.sum())
            cube[iy, ix] = np.clip(pat + D_true + 0.3 * rng.standard_normal((H, W)), 0, None)
    dc = fds.from_array(cube, q_per_px=0.02)
    vac = np.zeros((Sy, Sx), bool); vac[:, :Sx // 3] = True
    t_raw = thickness_map(dc, beam_radius=7.0)
    t, ex = thickness_map(dc, beam_radius=7.0, vacuum_mask=vac, return_extras=True)
    assert ex["dark"] == pytest.approx(D_true, abs=0.5)      # dark self-calibrated
    assert abs(t_raw[vac].mean()) > 1.0                       # dark fakes thick vacuum
    assert abs(t[vac].mean()) < 0.2                           # ...corrected to ~0
    corr = np.corrcoef(t[~vac].ravel(), tl_true[~vac].ravel())[0, 1]
    assert corr > 0.95                                        # recovers the ramp


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


# -- cepstral / FC-STEM (EWPC) ---------------------------------------------
def test_ewpc_calibration_cosine():
    # a cosine of m cycles across the pattern has a cepstral peak at r = m*dr,
    # dr = 1/(N*q_per_px). Verifies the quefrency (A) calibration.
    N, qpp = 128, 0.05
    dr = 1.0 / (N * qpp)
    yy, xx = np.mgrid[0:N, 0:N]
    for m in (8, 16, 24):
        I = 1.0 + 0.5 * np.cos(2 * np.pi * m * xx / N)
        cep = fds.ewpc_pattern(I, window=False)
        r, prof = fds.cepstral_radial_profile(cep, qpp, r_min=0.2)
        pk = r[np.argmax(prof)]
        assert abs(pk - m * dr) < 1.5 * (r[1] - r[0])       # within a bin
    assert fds.quefrency_per_px(N, qpp) == pytest.approx(dr)


def test_fluctuation_image_discriminates_order():
    from tests.synthetic import ring_pattern, bragg_pattern
    N, qpp = 96, 0.05
    Ry, Rx = 4, 6
    cube = np.empty((Ry, Rx, N, N))
    for iy in range(Ry):
        for ix in range(Rx):
            if ix < Rx // 2:                                # ordered: sharp spots
                cube[iy, ix] = bragg_pattern((N, N), spots=6, radius=22, amp=6) + 2
            else:                                            # disordered: halo
                cube[iy, ix] = ring_pattern((N, N), rings=((22, 4, 1.0),),
                                            central_beam=8.0) + 2
    dc = fds.from_array(cube, q_per_px=qpp)
    F = fds.fluctuation_image(dc, r_in=0.6, r_out=1.6, q_per_px=qpp, n_jobs=1)
    assert F.shape == (Ry, Rx)
    assert F[:, :Rx // 2].mean() > F[:, Rx // 2:].mean()     # ordered brighter


def test_ewpc_profiles_shape():
    from tests.synthetic import ring_pattern
    N = 64
    cube = np.stack([[ring_pattern((N, N), rings=((16, 3, 1.0),), central_beam=5.0)
                      for _ in range(3)] for _ in range(2)])
    dc = fds.from_array(cube, q_per_px=0.05)
    profs, r = fds.ewpc_profiles(dc, q_per_px=0.05, n_bins=24, n_jobs=1)
    assert profs.shape[0] == 6 and profs.shape[1] == r.size
    assert np.all(np.diff(r) > 0)


# -- material mask (exclude vacuum/empty positions) ------------------------
def test_material_mask_excludes_vacuum():
    from tests.synthetic import ring_pattern
    mat = ring_pattern((40, 40), rings=((12, 3, 1.0),), central_beam=4.0)
    vac = ring_pattern((40, 40), rings=(), central_beam=4.0)        # beam only, no ring
    Ry, Rx = 10, 8
    cube = np.empty((Ry, Rx, 40, 40))
    for iy in range(Ry):
        cube[iy, :] = vac if iy >= Ry - 3 else mat                  # bottom 3 rows empty
    cube += 0.02 * np.random.default_rng(0).standard_normal(cube.shape)
    dc = fds.from_array(cube, q_per_px=0.05)
    # automatic (Otsu)
    m = fds.material_mask(dc)
    assert m.shape == (Ry, Rx)
    assert m[:Ry - 3].mean() > 0.9         # material rows kept
    assert m[Ry - 3:].mean() < 0.1         # empty rows excluded
    # empty-region-anchored threshold
    empty = np.zeros((Ry, Rx), bool); empty[-3:, :] = True
    m2 = fds.material_mask(dc, empty_mask=empty)
    assert (~m2[-3:]).mean() > 0.8 and m2[:Ry - 3].all()


# -- aligned averaging (beam-wander correction) ----------------------------
def test_average_pattern_aligned_sharpens_wander():
    from tests.synthetic import ring_pattern
    from scipy.ndimage import shift as ndshift
    base = ring_pattern((64, 64), rings=((16, 2, 1.0),), central_beam=8.0)
    Ry, Rx = 6, 6
    cube = np.empty((Ry, Rx, 64, 64))
    rng = np.random.default_rng(0)
    for iy in range(Ry):
        for ix in range(Rx):
            dx, dy = rng.integers(-6, 7), rng.integers(-6, 7)   # random beam wander
            cube[iy, ix] = ndshift(base, (dy, dx), order=1, mode="nearest")
    dc = fds.from_array(cube, q_per_px=0.05)
    mask = np.ones((Ry, Rx), bool)
    plain = fds.average_pattern(dc, mask)
    aligned = fds.average_pattern_aligned(dc, mask, threshold=0.3)
    # aligned average has a sharper (higher-contrast) central beam than the smeared plain one
    assert aligned.max() > plain.max() * 1.2
    assert aligned.shape == (64, 64)


# -- fixed bad-pixel detection + repair -------------------------------------
def test_bad_pixel_map_and_repair():
    from tests.synthetic import ring_pattern
    base = ring_pattern((64, 64), rings=((18, 4, 1.0),), central_beam=6.0)
    Ry, Rx = 4, 5
    cube = np.tile(base, (Ry, Rx, 1, 1)).astype(float)
    cube += 0.05 * np.random.default_rng(0).standard_normal(cube.shape)
    cube[:, :, 10, 20] = 9999.0                     # fixed hot pixel every frame
    cube[:, :, 40, 45] = 8000.0
    dc = fds.from_array(cube, q_per_px=0.05)
    bad = fds.bad_pixel_map(dc.mean_dp(), hot_threshold=8.0)
    assert bad[10, 20] and bad[40, 45]              # hot pixels flagged
    assert bad.mean() < 0.02                        # structure NOT mass-flagged
    rep = fds.repair_bad_pixels(dc, bad)
    assert rep.data[:, :, 10, 20].mean() < 10       # repaired to neighbour level
    assert rep.metadata.get("bad_pixels_repaired") == int(bad.sum())


def test_unmix_nnls_recovers_pure_and_mixture():
    """NNLS unmixing IDs a pure compound and recovers a known mixing ratio."""
    r = np.arange(0.5, 8.0, 0.02)
    refs = fds.build_references(r, sigma=0.5)
    # pure Li2S -> almost all Li2S
    names, ab, res = fds.unmix_nnls(refs["Li2S"] * 3.0, refs, r=r, r_range=(1.2, 6.0))
    d = dict(zip(names, ab))
    assert d["Li2S"] > 0.9 and d["LiF"] < 0.1
    # 2:1 LiF:Li2CO3 (both separable) -> ratio recovered within tolerance
    mix = refs["LiF"] * 2.0 + refs["Li2CO3"] * 1.0
    names, ab, res = fds.unmix_nnls(mix, refs, r=r, r_range=(1.2, 6.0))
    d = dict(zip(names, ab))
    assert d["LiF"] > d["Li2CO3"] > 0.1
    assert abs(d["LiF"] / (d["Li2CO3"] + 1e-9) - 2.0) < 0.6
    assert res < 0.2                                  # good fit


def test_unmix_absent_compound_gets_zero():
    """A candidate that is not present receives ~zero abundance."""
    r = np.arange(0.5, 8.0, 0.02)
    refs = fds.build_references(r, sigma=0.5)
    names, ab, res = fds.unmix_nnls(refs["Li2CO3"] * 2.0, refs, r=r, r_range=(1.2, 6.0))
    d = dict(zip(names, ab))
    assert d["Li2S"] < 0.05                           # Li2S truly absent


def test_reference_degeneracy_flags_collinear_pairs():
    """~2.0 A compounds are collinear at low resolution; Li2S/Li2CO3 are not."""
    r = np.arange(0.5, 8.0, 0.02)
    refs = fds.build_references(r, sigma=0.5)
    names, C = fds.reference_degeneracy(refs)
    idx = {n: i for i, n in enumerate(names)}
    assert C[idx["LiF"], idx["Li2O"]] > 0.85          # hard to separate
    assert C[idx["Li2S"], idx["Li2CO3"]] < 0.3        # easily separated


def test_find_fsdp_finds_ring_over_beam_decay():
    """FSDP finder ignores the beam shoulder and locates a weak ring bump."""
    q = np.linspace(0.01, 1.1, 200)
    Iq = 200 * np.exp(-q / 0.08) + 5 * np.exp(-((q - 0.6) / 0.06) ** 2) + 0.5
    qp, conf = fds.find_fsdp(q, Iq)
    assert abs(qp - 0.6) < 0.05 and conf > 3.0
    # pure monotonic decay -> low confidence (no real ring)
    qp2, conf2 = fds.find_fsdp(q, 200 * np.exp(-q / 0.08) + 0.5)
    assert conf2 < conf                                # weaker than a real bump


def test_match_rings_identifies_large_d_ring():
    """A ring at d~4.1 A (q~0.244) matches Li2CO3, not the rocksalt/antifluorite set."""
    ranked = fds.match_rings([0.244], tol=0.03)
    top = ranked[0][0]
    assert top in ("Li2CO3", "Li3N")                 # only large-cell phases have a ring here
    scores = {c: s for c, s, *_ in ranked}
    assert scores["LiF"] == 0 and scores["Li2O"] == 0 and scores["Li2S"] == 0
    # exact-position winner is Li2CO3 (4.157 A -> q 0.2406)
    assert abs(1 / 4.157 - 0.244) < abs(1 / 3.872 - 0.244)


def test_synth_compound_iq_peaks_at_ring_positions():
    """The synthetic ring profile peaks at a compound's ring q=1/d positions."""
    q = np.linspace(0.1, 1.1, 400)
    iq = fds.synth_compound_iq("LiF", q, sigma_q=0.02)
    qpk = q[np.argmax(iq)]
    assert abs(qpk - 1 / 2.324) < 0.02          # LiF strongest ring d=2.324 A


def test_substrate_rings_explain_low_d_ring():
    """A d~1.50 A ring (q~0.665) is copper oxide (Cu2O/CuO), not a Li compound."""
    ranked = fds.match_rings([0.665], rings=fds.ALL_RINGS, tol=0.02)
    top = ranked[0][0]
    assert top in ("Cu2O", "CuO")
    # and the d=4.18 ring stays a Li compound (Cu phases cannot reach d>3.3)
    ranked2 = fds.match_rings([0.239], rings=fds.ALL_RINGS, tol=0.02)
    assert ranked2[0][0] in ("Li2CO3", "Li3N")
