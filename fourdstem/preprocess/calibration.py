"""
fourdstem.preprocess.calibration
================================
Reciprocal-space calibration helpers. Convert detector pixels to q, and
calibrate q_per_px from a known ring (e.g. a standard's d-spacing).
"""
from __future__ import annotations
import numpy as np


def q_per_px_from_ring(ring_radius_px, d_spacing_A):
    """Calibrate q_per_px (Å⁻¹/px) from a ring at ``ring_radius_px`` with known d.

    Uses the crystallographer convention q = 1/d.
    """
    q_ring = 1.0 / d_spacing_A
    return q_ring / ring_radius_px


def pixels_to_q(radius_px, q_per_px):
    """Convert a radius (px) to q (Å⁻¹)."""
    return np.asarray(radius_px, float) * q_per_px


def q_to_pixels(q, q_per_px):
    """Convert q (Å⁻¹) to a radius in pixels."""
    return np.asarray(q, float) / q_per_px


def q_axis(n_px, q_per_px):
    """Build a q axis (Å⁻¹) for ``n_px`` radial bins."""
    return np.arange(n_px, dtype=float) * q_per_px


def max_q_for_center(shape, center, q_per_px):
    """Largest q (Å⁻¹) fully sampled given the center's distance to each edge."""
    H, W = shape
    cx, cy = center
    return min(cx, cy, W - cx, H - cy) * q_per_px
