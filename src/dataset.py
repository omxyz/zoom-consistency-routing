"""ScreenSpot-Pro dataset handling."""

import json
import os
import random
from pathlib import Path
from PIL import Image

DATASET_ID = "likaixin/ScreenSpot-Pro"
DATA_DIR = Path("data/ScreenSpot-Pro")
SCREENING_SIZE = 200
SEED = 42


def download_dataset():
    """Download ScreenSpot-Pro from HuggingFace and return flat sample list."""
    from huggingface_hub import snapshot_download

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_DIR.exists() or not (DATA_DIR / "annotations").exists():
        print("Downloading ScreenSpot-Pro dataset...")
        snapshot_download(
            repo_id=DATASET_ID, repo_type="dataset",
            local_dir=str(DATA_DIR),
        )
        print("Download complete.")
    else:
        print("Dataset already cached.")

    samples = []
    annotations_dir = DATA_DIR / "annotations"
    for ann_file in sorted(annotations_dir.glob("*.json")):
        with open(ann_file) as f:
            items = json.load(f)
        for item in items:
            item["_img_dir"] = str(DATA_DIR / "images")
            samples.append(item)

    print(f"Loaded {len(samples)} samples from {len(list(annotations_dir.glob('*.json')))} annotation files.")
    return samples


def get_screening_subset(ds):
    """Return a fixed 200-sample subset for fast iteration."""
    indices = list(range(len(ds)))
    rng = random.Random(SEED)
    rng.shuffle(indices)
    return indices[:SCREENING_SIZE]


def get_sample(ds, idx):
    """Extract a single sample with image, instruction, bbox, and metadata."""
    item = ds[idx]
    img_path = os.path.join(item["_img_dir"], item["img_filename"])
    image = Image.open(img_path).convert("RGB")
    w, h = image.size

    raw_bbox = item.get("bbox")
    if raw_bbox and len(raw_bbox) == 4:
        x1, y1, x2, y2 = raw_bbox
        img_w, img_h = item.get("img_size", [w, h])
        bbox = [x1 / img_w, y1 / img_h, (x2 - x1) / img_w, (y2 - y1) / img_h]
    else:
        bbox = None

    return {
        "image": image,
        "instruction": item.get("instruction", ""),
        "bbox": bbox,
        "application": item.get("application", "unknown"),
        "group": item.get("group", "unknown"),
        "platform": item.get("platform", "unknown"),
        "ui_type": item.get("ui_type", "unknown"),
        "img_width": w,
        "img_height": h,
    }


def point_in_box(pred_x, pred_y, bbox):
    """Check if predicted point falls inside the bounding box. All values in [0, 1]."""
    if bbox is None or pred_x is None or pred_y is None:
        return False
    bx, by, bw, bh = bbox
    return (bx <= pred_x <= bx + bw) and (by <= pred_y <= by + bh)
