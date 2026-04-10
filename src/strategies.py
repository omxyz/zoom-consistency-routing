"""All routing strategies for offline analysis."""

import math

from src.dataset import point_in_box


def _pt(v):
    """Parse a prediction value from JSON (may be list, tuple, or None)."""
    if v is None:
        return None
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        try:
            return (float(v[0]), float(v[1]))
        except (ValueError, TypeError):
            return None
    return None


def _cons(s2):
    """Compute zoom consistency from step-2 prediction."""
    if s2 is None:
        return None
    try:
        dx = float(s2[0]) - 500.0
        dy = float(s2[1]) - 500.0
        return math.sqrt(dx * dx + dy * dy)
    except (ValueError, TypeError):
        return None


def _centroid(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def precompute(row):
    """Add derived fields to a data row for strategy evaluation."""
    row["_kv_final"] = _pt(row.get("kv_final"))
    row["_qw_final"] = _pt(row.get("qw_final"))
    row["_kv_cons"] = _cons(row.get("kv_s2"))
    row["_qw_cons"] = _cons(row.get("qw_s2_qwcrop"))
    row["_ss_final"] = _pt(row.get("qw_stage_split_final"))
    row["_kv_s1"] = _pt(row.get("kv_s1"))
    row["_qw_s1"] = _pt(row.get("qw_s1"))
    return row


# --- Strategy functions ---
# Each takes a precomputed row and returns a prediction (x, y) or None.


def kv_only(r):
    """KV-Ground baseline (both steps)."""
    return r["_kv_final"]


def qwen_only(r):
    """Qwen3.5-27B baseline (both steps)."""
    return r["_qw_final"]


def stage_split(r):
    """KV-Ground step-1, Qwen step-2 on KV's crop."""
    return r["_ss_final"]


def midpoint_fusion(r):
    """Average of both models' final predictions."""
    return _centroid(r["_kv_final"], r["_qw_final"])


def step1_vote(r):
    """Centroid of both models' step-1 predictions (no zoom)."""
    a, b = r["_kv_s1"], r["_qw_s1"]
    if a is None or b is None:
        return r["_kv_final"]
    c = ((a[0] + b[0]) / 2 / 1000.0, (a[1] + b[1]) / 2 / 1000.0)
    return (max(0, min(1, c[0])), max(0, min(1, c[1])))


def vote_agree(r, threshold=50):
    """If step-1 predictions agree within threshold, use centroid; else KV."""
    a, b = r["_kv_s1"], r["_qw_s1"]
    if a is None or b is None:
        return r["_kv_final"]
    d = math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    if d < threshold:
        c = ((a[0] + b[0]) / 2 / 1000.0, (a[1] + b[1]) / 2 / 1000.0)
        return (max(0, min(1, c[0])), max(0, min(1, c[1])))
    return r["_kv_final"]


def lower_consistency_router(r):
    """Pick the model with lower zoom consistency (more confident).
    This is the winning strategy."""
    kc, qc = r["_kv_cons"], r["_qw_cons"]
    if kc is None and qc is None:
        return r["_kv_final"]
    if kc is None:
        return r["_qw_final"]
    if qc is None:
        return r["_kv_final"]
    return r["_kv_final"] if kc <= qc else r["_qw_final"]


def bbox_norm_consistency(r):
    """Lower consistency, but normalized by bbox diagonal."""
    kc, qc = r["_kv_cons"], r["_qw_cons"]
    bbox = r.get("bbox")
    if bbox:
        diag = math.sqrt(bbox[2] ** 2 + bbox[3] ** 2) * 1000
        if diag > 1:
            if kc is not None:
                kc = kc / diag
            if qc is not None:
                qc = qc / diag
    if kc is None:
        return r["_qw_final"]
    if qc is None:
        return r["_kv_final"]
    return r["_kv_final"] if kc <= qc else r["_qw_final"]


def kv_fallback(r, threshold=250):
    """Use KV unless its consistency exceeds threshold, then Qwen."""
    kc = r["_kv_cons"]
    if kc is None or kc > threshold:
        return r["_qw_final"]
    return r["_kv_final"]


def oracle(r):
    """Perfect router (upper bound). Picks whichever is correct."""
    kv = r["_kv_final"]
    qw = r["_qw_final"]
    bbox = r.get("bbox")
    if kv and point_in_box(kv[0], kv[1], bbox):
        return kv
    return qw


def evaluate(data, pred_fn):
    """Evaluate a strategy on all samples. Returns (accuracy, n_correct, n_total)."""
    n = len(data)
    correct = 0
    for r in data:
        p = pred_fn(r)
        if p is not None and point_in_box(p[0], p[1], r.get("bbox")):
            correct += 1
    return correct / n if n else 0, correct, n


def all_strategies():
    """Return list of (name, function) for all strategies."""
    return [
        ("KV-only baseline", kv_only),
        ("Qwen-only baseline", qwen_only),
        ("Stage split KV->Qwen", stage_split),
        ("Midpoint fusion", midpoint_fusion),
        ("Step1 vote centroid", step1_vote),
        ("Vote agree T=50", lambda r: vote_agree(r, 50)),
        ("Vote agree T=100", lambda r: vote_agree(r, 100)),
        ("Lower-consistency router", lower_consistency_router),
        ("Bbox-norm consistency", bbox_norm_consistency),
        ("KV fallback T=150", lambda r: kv_fallback(r, 150)),
        ("KV fallback T=200", lambda r: kv_fallback(r, 200)),
        ("KV fallback T=250", lambda r: kv_fallback(r, 250)),
        ("KV fallback T=300", lambda r: kv_fallback(r, 300)),
        ("Oracle (upper bound)", oracle),
    ]
