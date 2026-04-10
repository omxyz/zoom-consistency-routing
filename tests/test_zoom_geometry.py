"""Opt-in tests for the zoom pipeline's geometric primitives.

Run with:  pytest --optional tests/test_zoom_geometry.py
"""

import math

import pytest

pytest.importorskip("PIL")

from src.zoom import CROP_RATIO, compute_crop_box, remap, zoom_consistency  # noqa: E402

pytestmark = pytest.mark.optional


# ---------- compute_crop_box ----------


def test_crop_box_centered_on_midpoint():
    x1, y1, x2, y2 = compute_crop_box(960, 540, 1920, 1080)
    assert (x2 - x1, y2 - y1) == (960, 540)
    assert (x1, y1, x2, y2) == (480, 270, 1440, 810)


def test_crop_box_clamps_at_top_left_corner():
    x1, y1, x2, y2 = compute_crop_box(0, 0, 1920, 1080)
    assert (x1, y1) == (0, 0)
    assert (x2 - x1, y2 - y1) == (960, 540)


def test_crop_box_clamps_at_bottom_right_corner():
    x1, y1, x2, y2 = compute_crop_box(1920, 1080, 1920, 1080)
    assert (x2, y2) == (1920, 1080)
    assert (x2 - x1, y2 - y1) == (960, 540)


def test_crop_box_near_left_edge_keeps_full_width():
    x1, y1, x2, y2 = compute_crop_box(10, 540, 1920, 1080)
    assert x1 == 0
    assert x2 - x1 == 960


def test_crop_box_non_square_image():
    x1, y1, x2, y2 = compute_crop_box(1500, 500, 3000, 1000)
    assert (x2 - x1, y2 - y1) == (1500, 500)


def test_crop_box_returns_integers():
    box = compute_crop_box(123.7, 456.2, 1920, 1080)
    assert all(isinstance(v, int) for v in box)


def test_crop_box_ratio_matches_constant():
    w, h = 1600, 900
    x1, y1, x2, y2 = compute_crop_box(800, 450, w, h)
    assert (x2 - x1) == int(w * CROP_RATIO)
    assert (y2 - y1) == int(h * CROP_RATIO)


# ---------- zoom_consistency ----------


def test_consistency_zero_at_center():
    assert zoom_consistency((500, 500)) == 0.0


def test_consistency_axis_symmetric():
    assert zoom_consistency((500, 600)) == zoom_consistency((600, 500))
    assert zoom_consistency((400, 500)) == zoom_consistency((500, 400))


def test_consistency_3_4_5_triangle():
    # dx=300, dy=400 → 500
    assert zoom_consistency((800, 900)) == pytest.approx(500.0)


def test_consistency_none_input():
    assert zoom_consistency(None) is None


def test_consistency_monotonic_with_distance():
    a = zoom_consistency((500, 510))
    b = zoom_consistency((500, 550))
    c = zoom_consistency((500, 600))
    assert a < b < c


# ---------- remap ----------


def test_remap_center_of_crop():
    # crop (100,100)-(500,500) in a 1000x1000 image
    out = remap((500, 500), (100, 100, 500, 500), 1000, 1000)
    assert out == pytest.approx((0.3, 0.3))


def test_remap_top_left_of_crop():
    out = remap((0, 0), (100, 100, 500, 500), 1000, 1000)
    assert out == pytest.approx((0.1, 0.1))


def test_remap_bottom_right_of_crop():
    out = remap((1000, 1000), (100, 100, 500, 500), 1000, 1000)
    assert out == pytest.approx((0.5, 0.5))


def test_remap_output_always_in_unit_interval():
    out = remap((1500, -200), (100, 100, 500, 500), 1000, 1000)
    assert 0.0 <= out[0] <= 1.0
    assert 0.0 <= out[1] <= 1.0


def test_remap_none_inputs():
    assert remap(None, (0, 0, 100, 100), 1000, 1000) is None
    assert remap((500, 500), None, 1000, 1000) is None
