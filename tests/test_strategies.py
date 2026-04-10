"""Opt-in tests for routing strategies. These lock in the SOTA routing logic.

Run with:  pytest --optional tests/test_strategies.py
"""

import pytest

pytest.importorskip("PIL")

from src.strategies import (  # noqa: E402
    _centroid,
    _cons,
    _pt,
    evaluate,
    kv_fallback,
    lower_consistency_router,
    oracle,
    precompute,
    vote_agree,
)

pytestmark = pytest.mark.optional


def _row(**kwargs):
    """Build a precomputed row with sensible defaults."""
    base = {
        "kv_final": kwargs.get("kv_final"),
        "qw_final": kwargs.get("qw_final"),
        "kv_s1": kwargs.get("kv_s1"),
        "qw_s1": kwargs.get("qw_s1"),
        "kv_s2": kwargs.get("kv_s2"),
        "qw_s2_qwcrop": kwargs.get("qw_s2_qwcrop"),
        "qw_stage_split_final": kwargs.get("qw_stage_split_final"),
        "bbox": kwargs.get("bbox"),
    }
    return precompute(base)


# ---------- _pt ----------


def test_pt_parses_list():
    assert _pt([0.3, 0.4]) == (0.3, 0.4)


def test_pt_parses_tuple_of_length_three():
    assert _pt((1, 2, 3)) == (1.0, 2.0)


def test_pt_none_and_garbage():
    assert _pt(None) is None
    assert _pt("oops") is None
    assert _pt([]) is None
    assert _pt([None, 0.5]) is None


# ---------- _cons ----------


def test_cons_matches_zoom_consistency_semantics():
    assert _cons((500, 500)) == 0.0
    assert _cons((500, 600)) == pytest.approx(100.0)


def test_cons_handles_none_and_invalid():
    assert _cons(None) is None
    assert _cons(("x", "y")) is None


# ---------- _centroid ----------


def test_centroid_average():
    assert _centroid((0.2, 0.4), (0.4, 0.8)) == pytest.approx((0.3, 0.6))


def test_centroid_with_none():
    assert _centroid(None, (0.1, 0.2)) == (0.1, 0.2)
    assert _centroid((0.1, 0.2), None) == (0.1, 0.2)
    assert _centroid(None, None) is None


# ---------- lower_consistency_router (THE SOTA PATH) ----------


def test_router_picks_kv_when_more_confident():
    r = _row(
        kv_final=[0.3, 0.3], qw_final=[0.7, 0.7],
        kv_s2=[508, 495],   # distance ≈ 9.4
        qw_s2_qwcrop=[560, 470],  # distance ≈ 67.1
    )
    assert lower_consistency_router(r) == (0.3, 0.3)


def test_router_picks_qwen_when_more_confident():
    r = _row(
        kv_final=[0.3, 0.3], qw_final=[0.7, 0.7],
        kv_s2=[700, 700],
        qw_s2_qwcrop=[505, 505],
    )
    assert lower_consistency_router(r) == (0.7, 0.7)


def test_router_tie_goes_to_kv():
    """Pin the <= tie-break; flipping to < would change this."""
    r = _row(
        kv_final=[0.3, 0.3], qw_final=[0.7, 0.7],
        kv_s2=[600, 500], qw_s2_qwcrop=[500, 600],
    )
    assert lower_consistency_router(r) == (0.3, 0.3)


def test_router_falls_back_when_kv_consistency_missing():
    r = _row(
        kv_final=[0.3, 0.3], qw_final=[0.7, 0.7],
        kv_s2=None, qw_s2_qwcrop=[505, 505],
    )
    assert lower_consistency_router(r) == (0.7, 0.7)


def test_router_falls_back_when_qwen_consistency_missing():
    r = _row(
        kv_final=[0.3, 0.3], qw_final=[0.7, 0.7],
        kv_s2=[505, 505], qw_s2_qwcrop=None,
    )
    assert lower_consistency_router(r) == (0.3, 0.3)


def test_router_both_none_returns_kv_final():
    r = _row(kv_final=[0.3, 0.3], qw_final=[0.7, 0.7])
    assert lower_consistency_router(r) == (0.3, 0.3)


# ---------- vote_agree ----------


def test_vote_agree_within_threshold_uses_centroid():
    r = _row(kv_s1=[400, 400], qw_s1=[420, 420], kv_final=[0.9, 0.9])
    out = vote_agree(r, threshold=50)
    assert out == pytest.approx((0.41, 0.41))


def test_vote_agree_outside_threshold_falls_back_to_kv():
    r = _row(kv_s1=[100, 100], qw_s1=[900, 900], kv_final=[0.5, 0.5])
    assert vote_agree(r, threshold=50) == (0.5, 0.5)


def test_vote_agree_clamps_to_unit_interval():
    r = _row(kv_s1=[1100, 1100], qw_s1=[1100, 1100], kv_final=[0.0, 0.0])
    out = vote_agree(r, threshold=50)
    assert 0.0 <= out[0] <= 1.0
    assert 0.0 <= out[1] <= 1.0


def test_vote_agree_missing_step1_falls_back():
    r = _row(kv_s1=None, qw_s1=[400, 400], kv_final=[0.2, 0.2])
    assert vote_agree(r) == (0.2, 0.2)


# ---------- kv_fallback ----------


def test_kv_fallback_under_threshold_uses_kv():
    r = _row(kv_s2=[550, 500], kv_final=[0.3, 0.3], qw_final=[0.7, 0.7])
    # consistency = 50 < 150
    assert kv_fallback(r, threshold=150) == (0.3, 0.3)


def test_kv_fallback_over_threshold_switches_to_qwen():
    r = _row(kv_s2=[800, 800], kv_final=[0.3, 0.3], qw_final=[0.7, 0.7])
    # consistency ≈ 424 > 150
    assert kv_fallback(r, threshold=150) == (0.7, 0.7)


def test_kv_fallback_none_consistency_uses_qwen():
    r = _row(kv_s2=None, kv_final=[0.3, 0.3], qw_final=[0.7, 0.7])
    assert kv_fallback(r) == (0.7, 0.7)


# ---------- oracle ----------


def test_oracle_returns_kv_when_correct():
    r = _row(
        kv_final=[0.5, 0.5], qw_final=[0.1, 0.1],
        bbox=[0.4, 0.4, 0.2, 0.2],
    )
    assert oracle(r) == (0.5, 0.5)


def test_oracle_returns_qwen_when_kv_wrong():
    r = _row(
        kv_final=[0.1, 0.1], qw_final=[0.5, 0.5],
        bbox=[0.4, 0.4, 0.2, 0.2],
    )
    assert oracle(r) == (0.5, 0.5)


# ---------- evaluate ----------


def test_evaluate_empty_list():
    acc, correct, n = evaluate([], lower_consistency_router)
    assert (acc, correct, n) == (0, 0, 0)


def test_evaluate_all_correct():
    data = [
        _row(kv_final=[0.5, 0.5], kv_s2=[500, 500],
             qw_s2_qwcrop=[700, 700], qw_final=[0.9, 0.9],
             bbox=[0.4, 0.4, 0.2, 0.2]),
        _row(kv_final=[0.3, 0.3], kv_s2=[500, 500],
             qw_s2_qwcrop=[700, 700], qw_final=[0.9, 0.9],
             bbox=[0.25, 0.25, 0.1, 0.1]),
    ]
    acc, correct, n = evaluate(data, lower_consistency_router)
    assert (acc, correct, n) == (1.0, 2, 2)


def test_evaluate_none_predictions_score_zero():
    data = [_row(bbox=[0.0, 0.0, 1.0, 1.0])]
    acc, correct, n = evaluate(data, lambda r: None)
    assert (acc, correct, n) == (0.0, 0, 1)
