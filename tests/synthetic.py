"""Synthetic 4D-STEM data generators for tests and examples (numpy only)."""
import numpy as np


def ring_pattern(shape=(128, 128), center=None, rings=((25, 8, 1.0), (45, 6, 0.5)),
                 central_beam=3.0, noise=0.0, seed=0):
    """A centro-symmetric diffraction pattern: central beam + Gaussian rings.

    ``rings`` is a list of ``(radius_px, width_px, amplitude)``.
    """
    rng = np.random.default_rng(seed)
    H, W = shape
    if center is None:
        center = (W / 2.0, H / 2.0)
    cx, cy = center
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    img = central_beam * np.exp(-(r ** 2) / (2 * 4.0 ** 2))
    for r0, w, amp in rings:
        img += amp * np.exp(-((r - r0) ** 2) / (2 * w ** 2))
    if noise:
        img = img + noise * rng.standard_normal(shape)
    return np.clip(img, 0, None)


def bragg_pattern(shape=(128, 128), center=None, spots=6, radius=45, amp=5.0,
                  seed=1):
    """A crystalline pattern: discrete Bragg spots on a ring (plus faint halo)."""
    rng = np.random.default_rng(seed)
    H, W = shape
    if center is None:
        center = (W / 2.0, H / 2.0)
    cx, cy = center
    img = ring_pattern(shape, center, rings=((radius, 10, 0.15),), central_beam=2.0)
    yy, xx = np.mgrid[0:H, 0:W]
    for k in range(spots):
        th = 2 * np.pi * k / spots + 0.1 * rng.standard_normal()
        sx, sy = cx + radius * np.cos(th), cy + radius * np.sin(th)
        img += amp * np.exp(-((xx - sx) ** 2 + (yy - sy) ** 2) / (2 * 1.5 ** 2))
    return img


def scan_cube(scan=(6, 5), dp=(96, 96), seed=2):
    """A 4D cube where the left half is amorphous and the right half crystalline.

    Good for testing virtual imaging and NMF component separation.
    """
    rng = np.random.default_rng(seed)
    Ry, Rx = scan
    cube = np.empty((Ry, Rx, *dp), float)
    for iy in range(Ry):
        for ix in range(Rx):
            if ix < Rx // 2:
                cube[iy, ix] = ring_pattern(dp, noise=0.05,
                                            seed=int(rng.integers(1e6)))
            else:
                cube[iy, ix] = bragg_pattern(dp, seed=int(rng.integers(1e6)))
    return cube
