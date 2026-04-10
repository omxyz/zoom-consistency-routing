"""Zoom pipeline and consistency computation."""

import math
from PIL import Image

from src.models import run_vlm

CROP_RATIO = 0.5


def compute_crop_box(abs_x, abs_y, img_w, img_h, crop_ratio=CROP_RATIO):
    """Compute a crop box centered on (abs_x, abs_y) in pixel coordinates."""
    crop_w = int(img_w * crop_ratio)
    crop_h = int(img_h * crop_ratio)
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


def zoom_consistency(step2):
    """Compute zoom consistency: distance of step-2 prediction from crop center.
    Lower = more confident (step-1 was accurate, step-2 barely moved)."""
    if step2 is None:
        return None
    dx = step2[0] - 500.0
    dy = step2[1] - 500.0
    return math.sqrt(dx * dx + dy * dy)


def remap(step2, crop_box, orig_w, orig_h):
    """Remap step-2 prediction (in 1000x1000 crop space) to normalized [0,1]."""
    if step2 is None or crop_box is None:
        return None
    x1, y1, x2, y2 = crop_box
    abs_x = x1 + (step2[0] / 1000.0) * (x2 - x1)
    abs_y = y1 + (step2[1] / 1000.0) * (y2 - y1)
    return (max(0.0, min(1.0, abs_x / orig_w)),
            max(0.0, min(1.0, abs_y / orig_h)))


def predict_2step(model_name, image, instruction):
    """Run full 2-step zoom pipeline for a single model.

    Returns:
        step1: (x, y) in 1000x1000 full-image space, or None
        crop_box: (x1, y1, x2, y2) in original pixel space, or None
        step2: (x, y) in 1000x1000 crop space, or None
        final: (x, y) normalized [0,1], or None
        consistency: float (distance from crop center), or None
    """
    orig_w, orig_h = image.size

    step1 = run_vlm(model_name, image, instruction)
    if step1 is None:
        return None, None, None, None, None

    abs_x = step1[0] / 1000.0 * orig_w
    abs_y = step1[1] / 1000.0 * orig_h
    x1, y1, x2, y2 = compute_crop_box(abs_x, abs_y, orig_w, orig_h)
    cropped = image.crop((x1, y1, x2, y2)).resize((orig_w, orig_h), Image.LANCZOS)

    step2 = run_vlm(model_name, cropped, instruction)
    cropped.close()

    if step2 is None:
        final = (max(0.0, min(1.0, abs_x / orig_w)),
                 max(0.0, min(1.0, abs_y / orig_h)))
        return step1, (x1, y1, x2, y2), None, final, None

    cons = zoom_consistency(step2)
    final = remap(step2, (x1, y1, x2, y2), orig_w, orig_h)

    return step1, (x1, y1, x2, y2), step2, final, cons
