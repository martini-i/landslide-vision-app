"""
gradcam.py — Visualize which regions of an image drive the model's stable/unstable prediction.
Run with: python gradcam.py [image_path ...]
With no arguments, runs on a default set of misclassified validation images.
Outputs: gradcam_output/<filename>_gradcam.png (heatmap overlay)
"""

import os
import sys
from PIL import Image
import model_utils

OUTPUT_DIR = "gradcam_output"

DEFAULT_IMAGES = [
    "slope_dataset/val/stable/stable_cliff_002.jpg",
    "slope_dataset/val/stable/stable_cliff_007.png",
    "slope_dataset/val/stable/stable_rock_001.jpg",
    "slope_dataset/val/stable/stable_dry_001.jpg",
    "slope_dataset/val/stable/stable_dry_003.jpg",
    "slope_dataset/val/stable/stable_engineered_009.jpg",
    "slope_dataset/val/unstable/unstable_erosion_011.jpg",
    "slope_dataset/val/unstable/unstable_scarp_001.jpg",
]

def run_gradcam(image_path):
    image = Image.open(image_path).convert("RGB")
    result = model_utils.predict(image)
    pred_class = max(result["scores"], key=result["scores"].get)
    pred_conf = result["scores"][pred_class]

    overlay = model_utils.gradcam_overlay(image)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(OUTPUT_DIR, f"{base}_gradcam.png")
    overlay.save(out_path)

    parent_dir = os.path.basename(os.path.dirname(image_path))
    true_label = "unstable" if parent_dir == "unstable" else "stable"
    print(f"{base:30s} true={true_label:9s} pred={pred_class:9s} conf={pred_conf:.2f}  -> {out_path}")

if __name__ == "__main__":
    images = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_IMAGES
    for img_path in images:
        if os.path.exists(img_path):
            run_gradcam(img_path)
        else:
            print(f"Skipping missing file: {img_path}")
