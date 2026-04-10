# Zoom Consistency Routing

**Confidence-Based Model Routing via Zoom Consistency for GUI Grounding**

A training-free ensemble method that achieves **80.9%** on ScreenSpot-Pro, establishing a new state of the art.

## Method

Two models run independent 2-step zoom-in pipelines on each sample. Per sample, whichever model's step-2 prediction is closer to the crop center (lower "zoom consistency" = higher confidence) is selected.

- **KV-Ground-8B** (specialist): selected for 64.9% of samples
- **Qwen3.5-27B** (generalist): selected for 35.1% of samples

No training required. The confidence signal is free — computed from values the zoom pipeline already produces.

## Results

| Method | ScreenSpot-Pro |
|---|---|
| **Zoom Consistency Router (ours)** | **80.9%** |
| KV-Ground-8B + ZoomIn (prev. SOTA) | 80.5% |
| KV-Ground-8B (our reproduction) | 80.1% |
| Qwen3.5-27B + ZoomIn | 60.9% |

## Setup

### Requirements

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install transformers peft huggingface_hub accelerate compressed-tensors autoawq
```

**Hardware**: NVIDIA H200 141GB (or any GPU with 80GB+ VRAM that can hold both models). Qwen3.5-27B-AWQ decompresses to bf16 in memory (~56GB).

### Download Models

```bash
python scripts/download_models.py
```

This downloads:
- `vocaela/KV-Ground-8B-BaseGuiOwl1.5-0315` → `models/kv-ground-8b/`
- `cyankiwi/Qwen3.5-27B-AWQ-4bit` → `models/qwen3.5-27b-awq/`

### Download Dataset

```bash
python scripts/download_dataset.py
```

Downloads ScreenSpot-Pro from HuggingFace to `data/ScreenSpot-Pro/`.

## Reproducing Results

### Step 1: Capture KV-Ground predictions

```bash
python run_capture.py --model kv --out results/kv_capture.json
```

### Step 2: Capture Qwen predictions + cross-model data

```bash
python run_capture.py --model qwen --kv-capture results/kv_capture.json --out results/qwen_cross_capture.json
```

### Step 3: Analyze all routing strategies

```bash
python analyze.py --input results/qwen_cross_capture.json
```

This prints accuracy for all strategies including the zoom consistency router.

### Step 4: Generate submission

```bash
python generate_submission.py --input results/qwen_cross_capture.json --out results/submission.json
```

## Quick Test (5 samples)

```bash
python run_capture.py --model kv --end 5 --out results/kv_test.json
python run_capture.py --model qwen --kv-capture results/kv_test.json --end 5 --out results/qwen_test.json
python analyze.py --input results/qwen_test.json
```

## Project Structure

```
zoom-consistency-routing/
├── README.md
├── paper.md                    # Full paper
├── requirements.txt
├── run_capture.py              # Main capture script (both models)
├── analyze.py                  # Offline analysis of all routing strategies
├── generate_submission.py      # Generate leaderboard submission JSON
├── src/
│   ├── __init__.py
│   ├── models.py               # Model loading and inference
│   ├── dataset.py              # ScreenSpot-Pro dataset handling
│   ├── zoom.py                 # Zoom pipeline and consistency computation
│   └── strategies.py           # All routing strategies
├── scripts/
│   ├── download_models.py      # Download both models
│   └── download_dataset.py     # Download ScreenSpot-Pro
└── results/
    └── submission.json         # Our submission (80.9%)
```

## Citation

```bibtex
@article{zoomconsistency2026,
  title={Confidence-Based Model Routing via Zoom Consistency for GUI Grounding},
  author={},
  year={2026}
}
```

## License

MIT
