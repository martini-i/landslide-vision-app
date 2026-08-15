"""
One-time preprocessing: applies model_utils.crop_sky() to every image in
slope_dataset/{train,val}/{stable,unstable}, overwriting them in place.

Must be run once whenever crop_sky's logic changes, so training and
inference stay consistent (the model must be retrained afterward via
train_model.py). Dataset is git-tracked, so this is recoverable via git
if something goes wrong.
"""

from pathlib import Path
from PIL import Image
from model_utils import crop_sky

DATA_DIR = Path("slope_dataset")
EXTENSIONS = {".jpg", ".jpeg", ".png"}

paths = sorted(
    p for p in DATA_DIR.glob("*/*/*")
    if p.suffix.lower() in EXTENSIONS
)

cropped_count = 0
for i, p in enumerate(paths, start=1):
    img = Image.open(p).convert("RGB")
    cropped = crop_sky(img)
    if cropped.height != img.height:
        ratio = cropped.height / img.height
        print(f"[{i}/{len(paths)}] cropped {ratio:.2f}  {p.relative_to(DATA_DIR)}")
        cropped.save(p)
        cropped_count += 1
    if i % 20 == 0:
        print(f"...{i}/{len(paths)} processed")

print(f"\nDone. {cropped_count}/{len(paths)} images cropped and overwritten.")
