"""Download ScreenSpot-Pro dataset."""
from src.dataset import download_dataset

ds = download_dataset()
print(f"Dataset ready: {len(ds)} samples")
