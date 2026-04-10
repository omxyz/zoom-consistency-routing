"""Download both models to models/ directory."""
from huggingface_hub import snapshot_download

print("Downloading KV-Ground-8B...", flush=True)
snapshot_download(
    repo_id="vocaela/KV-Ground-8B-BaseGuiOwl1.5-0315",
    local_dir="models/kv-ground-8b",
    max_workers=8,
)
print("KV-Ground-8B done.", flush=True)

print("Downloading Qwen3.5-27B-AWQ-4bit...", flush=True)
snapshot_download(
    repo_id="cyankiwi/Qwen3.5-27B-AWQ-4bit",
    local_dir="models/qwen3.5-27b-awq",
    max_workers=8,
)
print("Qwen3.5-27B-AWQ done.", flush=True)
print("All models downloaded.", flush=True)
