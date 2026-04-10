"""
analyze.py - Evaluate all routing strategies offline from captured predictions.

Usage: python analyze.py --input results/qwen_cross_capture.json
"""
import argparse
import json
import math
import sys

from src.strategies import precompute, all_strategies, evaluate, lower_consistency_router
from src.dataset import point_in_box


def consistency_correlation(data, model_prefix, cons_key):
    """Show accuracy bucketed by zoom consistency."""
    buckets = {"<30": [], "30-80": [], "80-150": [], "150-250": [], "250+": []}
    for r in data:
        c = r.get(cons_key)
        final = r.get(f"_{model_prefix}_final")
        if c is None or final is None:
            continue
        correct = point_in_box(final[0], final[1], r.get("bbox"))
        if c < 30:
            buckets["<30"].append(correct)
        elif c < 80:
            buckets["30-80"].append(correct)
        elif c < 150:
            buckets["80-150"].append(correct)
        elif c < 250:
            buckets["150-250"].append(correct)
        else:
            buckets["250+"].append(correct)
    print(f"\n{model_prefix.upper()} consistency vs accuracy:")
    for k, items in buckets.items():
        if items:
            print("  %-10s n=%-5d acc=%.3f" % (k, len(items), sum(items) / len(items)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    args = ap.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    print("Loaded %d samples\n" % len(data))

    # Precompute derived fields
    for r in data:
        precompute(r)

    # Run all strategies
    strategies = all_strategies()
    print("%-30s %8s %8s %8s" % ("Strategy", "Acc", "Correct", "vs KV"))
    print("-" * 60)

    kv_base = None
    for label, fn in strategies:
        acc, correct, n = evaluate(data, fn)
        if label == "KV-only baseline":
            kv_base = acc
        delta = "%+.4f" % (acc - kv_base) if kv_base is not None else ""
        print("%-30s %8.4f %8d %8s" % (label, acc, correct, delta))

    # Consistency correlation
    consistency_correlation(data, "kv", "_kv_cons")
    consistency_correlation(data, "qw", "_qw_cons")

    # Router picks
    kv_picks = 0
    qw_picks = 0
    for r in data:
        pred = lower_consistency_router(r)
        kv = r["_kv_final"]
        if pred == kv:
            kv_picks += 1
        else:
            qw_picks += 1
    n = len(data)
    print("\nRouter picks: KV=%d (%.1f%%) Qwen=%d (%.1f%%)" % (
        kv_picks, 100 * kv_picks / n, qw_picks, 100 * qw_picks / n))


if __name__ == "__main__":
    main()
