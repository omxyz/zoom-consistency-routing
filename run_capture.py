"""
run_capture.py - Capture predictions from one or both models on ScreenSpot-Pro.

Usage:
  # Step 1: KV-Ground only
  python run_capture.py --model kv --out results/kv_capture.json

  # Step 2: Qwen + cross-model (reads KV capture for stage split)
  python run_capture.py --model qwen --kv-capture results/kv_capture.json --out results/qwen_cross_capture.json
"""
import argparse
import json
import os
import sys

from src.dataset import download_dataset, get_sample, point_in_box
from src.models import load_model, run_vlm
from src.zoom import predict_2step, compute_crop_box, remap, CROP_RATIO
from PIL import Image


def parse_maybe_tuple(v):
    if v is None:
        return None
    if isinstance(v, list) and len(v) >= 2:
        try:
            return [float(v[0]), float(v[1])]
        except (ValueError, TypeError):
            return None
    return None


def run_kv_capture(ds, indices, out_path):
    """Capture KV-Ground predictions on all samples."""
    load_model("kv")
    results = []
    correct_count = 0

    for i, idx in enumerate(indices):
        sample = get_sample(ds, idx)
        if sample["bbox"] is None:
            sample["image"].close()
            continue

        try:
            s1, crop_box, s2, final, cons = predict_2step("kv", sample["image"], sample["instruction"])
        except Exception as e:
            print(f"  [{i+1}] error idx={idx}: {e}", flush=True)
            s1 = crop_box = s2 = final = cons = None

        correct = final is not None and point_in_box(final[0], final[1], sample["bbox"])
        if correct:
            correct_count += 1

        results.append({
            "idx": idx,
            "instruction": sample["instruction"],
            "application": sample.get("application", "?"),
            "ui_type": sample.get("ui_type", "?"),
            "group": sample.get("group", "?"),
            "platform": sample.get("platform", "?"),
            "img_width": sample["img_width"],
            "img_height": sample["img_height"],
            "bbox": sample["bbox"],
            "step1": s1,
            "crop_box": crop_box,
            "step2": s2,
            "final": final,
            "correct": correct,
        })

        sample["image"].close()

        if (i + 1) % 25 == 0:
            with open(out_path, "w") as f:
                json.dump(results, f, default=str)
            acc = correct_count / (i + 1)
            print(f"  [{i+1}/{len(indices)}] acc={acc:.4f} correct={correct_count}", flush=True)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    final_acc = correct_count / len(results) if results else 0
    print(f"\nDone. {len(results)} samples. Accuracy: {final_acc:.4f}", flush=True)


def run_qwen_cross_capture(ds, indices, kv_capture_path, out_path, model_name="qwen"):
    """Capture generalist predictions + stage split using existing KV capture data."""
    with open(kv_capture_path) as f:
        kv_data = json.load(f)
    kv_by_idx = {r["idx"]: r for r in kv_data}
    print(f"Loaded {len(kv_data)} KV rows from {kv_capture_path}", flush=True)

    load_model(model_name)
    results = []
    n_correct_qwen = 0
    n_correct_stage = 0

    for i, idx in enumerate(indices):
        sample = get_sample(ds, idx)
        if sample["bbox"] is None:
            sample["image"].close()
            continue

        image = sample["image"]
        instruction = sample["instruction"]
        bbox = sample["bbox"]
        orig_w, orig_h = image.size

        kv_row = kv_by_idx.get(idx)
        if kv_row is None:
            print(f"  [{i+1}] no KV data for idx={idx}, skipping", flush=True)
            image.close()
            continue

        kv_s1 = parse_maybe_tuple(kv_row.get("step1"))
        kv_crop_box = kv_row.get("crop_box")
        kv_s2 = parse_maybe_tuple(kv_row.get("step2"))
        kv_final = parse_maybe_tuple(kv_row.get("final"))
        kv_correct = kv_row.get("correct", False)

        # Qwen step 1
        try:
            qw_s1 = run_vlm(model_name, image, instruction)
        except Exception as e:
            print(f"  [{i+1}] qwen s1 error: {e}", flush=True)
            qw_s1 = None

        # Qwen step 2 on own crop
        qw_crop = None
        qw_crop_box = None
        if qw_s1 is not None:
            abs_x = qw_s1[0] / 1000.0 * orig_w
            abs_y = qw_s1[1] / 1000.0 * orig_h
            x1, y1, x2, y2 = compute_crop_box(abs_x, abs_y, orig_w, orig_h)
            qw_crop = image.crop((x1, y1, x2, y2)).resize((orig_w, orig_h), Image.LANCZOS)
            qw_crop_box = (x1, y1, x2, y2)

        try:
            qw_s2_qwcrop = run_vlm(model_name, qw_crop, instruction) if qw_crop is not None else None
        except Exception as e:
            qw_s2_qwcrop = None

        # Qwen step 2 on KV's crop (stage split)
        qw_s2_kvcrop = None
        kv_crop_box_for_remap = None
        if kv_s1 is not None:
            kv_abs_x = kv_s1[0] / 1000.0 * orig_w
            kv_abs_y = kv_s1[1] / 1000.0 * orig_h
            kx1, ky1, kx2, ky2 = compute_crop_box(kv_abs_x, kv_abs_y, orig_w, orig_h)
            kv_crop_img = image.crop((kx1, ky1, kx2, ky2)).resize((orig_w, orig_h), Image.LANCZOS)
            kv_crop_box_for_remap = (kx1, ky1, kx2, ky2)
            try:
                qw_s2_kvcrop = run_vlm(model_name, kv_crop_img, instruction)
            except Exception as e:
                qw_s2_kvcrop = None
            kv_crop_img.close()

        # Compute finals
        qw_final = remap(qw_s2_qwcrop, qw_crop_box, orig_w, orig_h) if qw_s2_qwcrop else None
        if qw_final is None and qw_s1 is not None:
            qw_final = (max(0.0, min(1.0, qw_s1[0] / 1000.0)),
                        max(0.0, min(1.0, qw_s1[1] / 1000.0)))

        qw_stage_split_final = remap(qw_s2_kvcrop, kv_crop_box_for_remap, orig_w, orig_h) if qw_s2_kvcrop else None
        if qw_stage_split_final is None and kv_s1 is not None:
            qw_stage_split_final = (max(0.0, min(1.0, kv_s1[0] / 1000.0)),
                                     max(0.0, min(1.0, kv_s1[1] / 1000.0)))

        qw_correct = qw_final is not None and point_in_box(qw_final[0], qw_final[1], bbox)
        qw_ss_correct = qw_stage_split_final is not None and point_in_box(
            qw_stage_split_final[0], qw_stage_split_final[1], bbox)

        if qw_correct:
            n_correct_qwen += 1
        if qw_ss_correct:
            n_correct_stage += 1

        if qw_crop is not None:
            qw_crop.close()
        image.close()

        results.append({
            "idx": idx,
            "instruction": instruction,
            "application": sample.get("application", "?"),
            "ui_type": sample.get("ui_type", "?"),
            "group": sample.get("group", "?"),
            "platform": sample.get("platform", "?"),
            "img_width": orig_w,
            "img_height": orig_h,
            "bbox": bbox,
            "kv_s1": kv_s1, "kv_crop_box": kv_crop_box, "kv_s2": kv_s2,
            "kv_final": kv_final, "kv_correct": kv_correct,
            "qw_s1": qw_s1, "qw_crop_box": qw_crop_box,
            "qw_s2_qwcrop": qw_s2_qwcrop, "qw_final": qw_final, "qw_correct": qw_correct,
            "qw_s2_kvcrop": qw_s2_kvcrop,
            "qw_stage_split_final": qw_stage_split_final,
            "qw_stage_split_correct": qw_ss_correct,
        })

        if (i + 1) % 25 == 0:
            with open(out_path, "w") as f:
                json.dump(results, f, default=str)
            qw_acc = n_correct_qwen / (i + 1)
            ss_acc = n_correct_stage / (i + 1)
            print(f"  [{i+1}/{len(indices)}] qw_acc={qw_acc:.4f} stage_split_acc={ss_acc:.4f}", flush=True)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDone. {len(results)} samples. Qwen: {n_correct_qwen}/{len(results)}. Stage split: {n_correct_stage}/{len(results)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["kv", "qwen", "phi4"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--kv-capture", default=None, help="Path to KV capture JSON (required for --model qwen)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    args = ap.parse_args()

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    ds = download_dataset()
    end = args.end if args.end is not None else len(ds)
    indices = list(range(args.start, min(end, len(ds))))
    print(f"Running model={args.model} on {len(indices)} samples [{args.start}, {end})", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.model == "kv":
        run_kv_capture(ds, indices, args.out)
    elif args.model in ("qwen", "phi4"):
        if args.kv_capture is None:
            print(f"ERROR: --kv-capture required for --model {args.model}", file=sys.stderr)
            sys.exit(1)
        run_qwen_cross_capture(ds, indices, args.kv_capture, args.out,
                               model_name=args.model)


if __name__ == "__main__":
    main()
