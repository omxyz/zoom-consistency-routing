"""Opt-in tests for dataset helpers.

Run with:  pytest --optional tests/test_dataset.py
"""

import pytest

pytest.importorskip("PIL")

from src.dataset import point_in_box  # noqa: E402

pytestmark = pytest.mark.optional


def test_point_inside_box():
    assert point_in_box(0.5, 0.5, [0.4, 0.4, 0.2, 0.2]) is True


def test_point_outside_box():
    assert point_in_box(0.1, 0.1, [0.4, 0.4, 0.2, 0.2]) is False


def test_point_on_edge_counts_as_inside():
    assert point_in_box(0.4, 0.5, [0.4, 0.4, 0.2, 0.2]) is True
    assert point_in_box(0.6, 0.6, [0.4, 0.4, 0.2, 0.2]) is True


def test_zero_area_box_only_matches_exact_point():
    assert point_in_box(0.5, 0.5, [0.5, 0.5, 0.0, 0.0]) is True
    assert point_in_box(0.51, 0.5, [0.5, 0.5, 0.0, 0.0]) is False


def test_none_bbox_returns_false():
    assert point_in_box(0.5, 0.5, None) is False


def test_none_prediction_returns_false():
    assert point_in_box(None, 0.5, [0.4, 0.4, 0.2, 0.2]) is False
    assert point_in_box(0.5, None, [0.4, 0.4, 0.2, 0.2]) is False
