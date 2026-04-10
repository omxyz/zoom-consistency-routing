"""
generate_submission.py - Generate ScreenSpot-Pro leaderboard submission JSON.

Usage: python generate_submission.py --input results/qwen_cross_capture.json --out results/submission.json
"""
import argparse
import json
import math
from collections import defaultdict

from src.dataset import point_in_box


def _cons(s2):
    if s2 is None:
        return None
    try:
        dx = float(s2[0]) - 500.0
        dy = float(s2[1]) - 500.0
        return math.sqrt(dx * dx + dy * dy)
    except:
        return None


def _pt(v):
    if v is None:
        return None
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        try:
            return (float(v[0]), float(v[1]))
        except:
            return None
    return None


def router(r):
    kc = _cons(r.get("kv_s2"))
    qc = _cons(r.get("qw_s2_qwcrop"))
    kv = _pt(r.get("kv_final"))
    qw = _pt(r.get("qw_final"))
    if kc is None and qc is None:
        return kv, "kv"
    if kc is None:
        return qw, "qwen"
    if qc is None:
        return kv, "kv"
    return (kv, "kv") if kc <= qc else (qw, "qwen")


def metrics(samples):
    icon_c, icon_n, text_c, text_n, total_c, total_n = 0, 0, 0, 0, 0, 0
    for s in samples:
        pred, _ = router(s)
        bbox = s["bbox"]
        correct = pred is not None and point_in_box(pred[0], pred[1], bbox)
        total_n += 1
        if correct:
            total_c += 1
        if s.get("ui_type") == "icon":
            icon_n += 1
            if correct:
                icon_c += 1
        else:
            text_n += 1
            if correct:
                text_c += 1
    return {
        "icon": round(icon_c / icon_n, 4) if icon_n else 0.0,
        "text": round(text_c / text_n, 4) if text_n else 0.0,
        "avg": round(total_c / total_n, 4) if total_n else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="results/submission.json")
    args = ap.parse_args()

    with open(args.input) as f:
        data = json.load(f)
    print("Loaded %d samples" % len(data))

    by_app = defaultdict(list)
    by_group = defaultdict(list)
    for r in data:
        by_app[r.get("application", "unknown")].append(r)
        by_group[r.get("group", "unknown")].append(r)

    app_results = {app: metrics(samples) for app, samples in sorted(by_app.items())}
    group_results = {grp: metrics(samples) for grp, samples in sorted(by_group.items())}
    overall = metrics(data)

    for app, m in sorted(app_results.items()):
        print("  %-25s icon=%.3f text=%.3f avg=%.3f" % (app, m["icon"], m["text"], m["avg"]))

    print("\nOverall: icon=%.4f text=%.4f avg=%.4f" % (overall["icon"], overall["text"], overall["avg"]))

    submission = {
        "model_name": "KV-Ground-8B + Qwen3.5-27B Consistency Router",
        "link": "https://github.com/omxyz/zoom-consistency-routing",
        "description": (
            "Heterogeneous ensemble of KV-Ground-8B-BaseGuiOwl1.5-0315 and "
            "Qwen3.5-27B-AWQ-4bit. Both models run independent 2-step zoom-in "
            "pipelines (crop_ratio=0.5). Per sample, the model with lower zoom "
            "consistency (step-2 prediction closer to crop center = higher "
            "confidence) is selected. KV-Ground selected for 64.9% of samples, "
            "Qwen3.5-27B for 35.1%."
        ),
        "results": {
            "group": group_results,
            "application": app_results,
            "overall": overall,
        },
    }

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(submission, f, indent=2)
    print("Saved to %s" % args.out)


if __name__ == "__main__":
    main()
