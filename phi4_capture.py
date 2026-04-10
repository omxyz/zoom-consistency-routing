"""
phi4_capture.py - Capture Phi-4-Reasoning-Vision-15B predictions on ScreenSpot-Pro.
Reuses existing KV capture data.

Usage:
  python phi4_capture.py --end 5    # quick smoke test
  python phi4_capture.py             # full 1581 samples
"""
import argparse
import json
import math
import os
import re
import sys
sys.path.insert(0, "/workspace")

import torch
from PIL import Image
from prepare import download_dataset, get_sample, point_in_box

PHI4_PATH = os.environ.get("PHI4_PATH", "/workspace/phi4-vision-15b")
CROP_RATIO = 0.5
DEVICE = "cuda"

SYSTEM_PROMPT = (
    "You are a helpful assistant. The user will give you an instruction, "
    "and you MUST left click on the corresponding UI element via tool call. "
    "If you are not sure about where to click, guess a most likely one.\n\n"
    "# Tools\n\n"
    "You may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n"
    '{"type": "function", "function": {"name": "computer_use", '
    '"description": "Use a mouse to interact with a computer.\\n'
    "* The screen's resolution is 1000x1000.\\n"
    "* Make sure to click any buttons, links, icons, etc with the cursor tip "
    "in the center of the element. \\n"
    '* You can only use the left_click action to interact with the computer.", '
    '"parameters": {"properties": {"action": {"description": '
    '"The action to perform. The available actions are:\\n'
    '* `left_click`: Click the left mouse button with coordinate (x, y).", '
    '"enum": ["left_click"], "type": "string"}, '
    '"coordinate": {"description": "(x, y): The x (pixels from the left edge) '
    "and y (pixels from the top edge) coordinates to move the mouse to. "
    'Required only by `action=left_click`.", "type": "array"}, '
    '"required": ["action"], "type": "object"}}}}\n'
    "</tools>\n\n"
    "For each function call, return a json object with function name and "
    "arguments within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n'
    "</tool_call>"
)

_model = None
_processor = None


def load_phi4():
    global _model, _processor
    if _model is not None:
        return
    from transformers import AutoModelForCausalLM, AutoProcessor
    print(f"Loading Phi-4 from {PHI4_PATH}...", flush=True)
    _processor = AutoProcessor.from_pretrained(PHI4_PATH, trust_remote_code=True)
    _model = AutoModelForCausalLM.from_pretrained(
        PHI4_PATH, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map=DEVICE,
    )
    _model.eval()
    print("Phi-4 loaded.", flush=True)


def run_phi4(image, instruction):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"<image>\n{instruction}"},
    ]
    prompt = _processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = _processor(text=prompt, images=[image], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output_ids = _model.generate(**inputs, max_new_tokens=512, do_sample=False)
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    response = _processor.tokenizer.decode(generated, skip_special_tokens=True)
    return parse_tool_call(response)


def parse_tool_call(response):
    if not response:
        return None
    tc_match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", response, re.DOTALL)
    if tc_match:
        try:
            data = json.loads(tc_match.group(1))
            coord = data.get("arguments", {}).get("coordinate", [])
            if len(coord) >= 2:
                return (float(coord[0]), float(coord[1]))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    coord_match = re.search(
        r'"coordinate"\s*:\s*\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]', response)
    if coord_match:
        return (float(coord_match.group(1)), float(coord_match.group(2)))
    nums = re.findall(r'(\d{1,4}(?:\.\d+)?)', response)
    if len(nums) >= 2:
        x, y = float(nums[0]), float(nums[1])
        if 0 <= x <= 1000 and 0 <= y <= 1000:
            return (x, y)
    return None


def compute_crop_box(abs_x, abs_y, img_w, img_h):
    crop_w = int(img_w * CROP_RATIO)
    crop_h = int(img_h * CROP_RATIO)
    cx, cy = int(abs_x), int(abs_y)
    x1 = max(0, cx - crop_w // 2)
    y1 = max(0, cy - crop_h // 2)
    x2 = min(img_w, x1 + crop_w)
    y2 = min(img_h, y1 + crop_h)
    if x2 - x1 < crop_w:
        x1 = max(0, x2 - crop_w)
    if y2 - y1 < crop_h:
        y1 = max(0, y2 - crop_h)
    return x1, y1, x2, y2


def remap(step2, crop_box, orig_w, orig_h):
    if step2 is None or crop_box is None:
        return None
    x1, y1, x2, y2 = crop_box
    abs_x = x1 + (step2[0] / 1000.0) * (x2 - x1)
    abs_y = y1 + (step2[1] / 1000.0) * (y2 - y1)
    return (max(0.0, min(1.0, abs_x / orig_w)),
            max(0.0, min(1.0, abs_y / orig_h)))


def safe_run(image, instruction):
    try:
        return run_phi4(image, instruction)
    except Exception as e:
        print(f"  ! phi4 error: {e}", flush=True)
        return None


def parse_maybe_tuple(v):
    if v is None:
        return None
    if isinstance(v, list) and len(v) >= 2:
        try:
            return [float(v[0]), float(v[1])]
        except (ValueError, TypeError):
            return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--kv-capture", default="/workspace/kv_capture.json")
    ap.add_argument("--out", default="/workspace/phi4_cross_capture.json")
    args = ap.parse_args()

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    with open(args.kv_capture) as f:
        kv_data = json.load(f)
    kv_by_idx = {r["idx"]: r for r in kv_data}
    print(f"Loaded {len(kv_data)} KV rows", flush=True)

    ds = download_dataset()
    end = args.end if args.end is not None else len(ds)
    indices = list(range(args.start, min(end, len(ds))))
    print(f"Running Phi-4 on {len(indices)} samples [{args.start}, {end})", flush=True)

    load_phi4()

    results = []
    n_correct = 0

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
            image.close()
            continue

        kv_s1 = parse_maybe_tuple(kv_row.get("step1"))
        kv_s2 = parse_maybe_tuple(kv_row.get("step2"))
        kv_final = parse_maybe_tuple(kv_row.get("final"))
        kv_correct = kv_row.get("correct", False)
        kv_crop_box = kv_row.get("crop_box")

        phi_s1 = safe_run(image, instruction)

        phi_crop_box = None
        phi_s2 = None
        if phi_s1 is not None:
            abs_x = phi_s1[0] / 1000.0 * orig_w
            abs_y = phi_s1[1] / 1000.0 * orig_h
            x1, y1, x2, y2 = compute_crop_box(abs_x, abs_y, orig_w, orig_h)
            phi_crop_box = (x1, y1, x2, y2)
            phi_crop = image.crop((x1, y1, x2, y2)).resize((orig_w, orig_h), Image.LANCZOS)
            phi_s2 = safe_run(phi_crop, instruction)
            phi_crop.close()

        phi_final = remap(phi_s2, phi_crop_box, orig_w, orig_h)
        if phi_final is None and phi_s1 is not None:
            phi_final = (max(0.0, min(1.0, phi_s1[0] / 1000.0)),
                         max(0.0, min(1.0, phi_s1[1] / 1000.0)))

        phi_correct = phi_final is not None and point_in_box(phi_final[0], phi_final[1], bbox)
        if phi_correct:
            n_correct += 1

        image.close()

        results.append({
            "idx": idx,
            "instruction": instruction,
            "application": sample.get("application", "?"),
            "ui_type": sample.get("ui_type", "?"),
            "group": sample.get("group", "?"),
            "platform": sample.get("platform", "?"),
            "img_width": orig_w, "img_height": orig_h, "bbox": bbox,
            "kv_s1": kv_s1, "kv_crop_box": kv_crop_box, "kv_s2": kv_s2,
            "kv_final": kv_final, "kv_correct": kv_correct,
            "qw_s1": phi_s1, "qw_crop_box": phi_crop_box,
            "qw_s2_qwcrop": phi_s2, "qw_final": phi_final, "qw_correct": phi_correct,
        })

        if (i + 1) % 5 == 0:
            with open(args.out, "w") as f:
                json.dump(results, f, default=str)
            acc = n_correct / (i + 1)
            print(f"  [{i+1}/{len(indices)}] phi4_acc={acc:.4f} correct={n_correct}", flush=True)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    final_acc = n_correct / len(results) if results else 0
    print(f"\nDone. {len(results)} samples. Phi-4 accuracy: {final_acc:.4f} ({n_correct}/{len(results)})", flush=True)


if __name__ == "__main__":
    main()
