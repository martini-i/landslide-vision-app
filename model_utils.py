"""
model_utils.py — Shared model loading, prediction, and Grad-CAM logic.
Used by app.py (Gradio), backend/main.py (FastAPI), and gradcam.py (CLI).
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms, models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ADE20K class index for "sky" in the pretrained segmentation model used by crop_sky().
SKY_CLASS_INDEX = 2
SKY_PIXEL_MIN_FRACTION = 0.03   # below this, treat as no real sky band present
SKY_CROP_MAX_FRACTION = 0.55    # never crop away more than this much of the image

MODEL_PATH = "slope_model.pth"

# Decision threshold is intentionally lower than 0.5: missing a genuinely unstable
# slope (false negative) is a costlier error than a false alarm, so "Potentially
# Unstable" is flagged starting at P(unstable) >= 0.35 rather than waiting for it
# to be the argmax.
UNSTABLE_THRESHOLD = 0.35
UNSTABLE_HIGH_CONFIDENCE = 0.65

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

_model = None
_class_names = None
_device = None
_cam = None
_seg_processor = None
_seg_model = None


def load_model():
    """Load the trained checkpoint once and cache it. Safe to call repeatedly."""
    global _model, _class_names, _device, _cam
    if _model is not None:
        return _model, _class_names, _device

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=_device, weights_only=False)
    _class_names = checkpoint["class_names"]
    arch = checkpoint.get("model_arch", "ResNet")

    if "EfficientNet" in arch:
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, 2))
        target_layer = model.features[-1]
    else:
        model = models.resnet18(weights=None)
        model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, 2))
        target_layer = model.layer4[-1]

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(_device)
    model.eval()

    _model = model
    _cam = GradCAM(model=model, target_layers=[target_layer])
    return _model, _class_names, _device


def _load_segmentation_model():
    """Load the pretrained sky-segmentation model once and cache it."""
    global _seg_processor, _seg_model
    if _seg_model is not None:
        return _seg_processor, _seg_model

    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
    _seg_processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    _seg_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    _seg_model.eval()
    return _seg_processor, _seg_model


def crop_sky(image: Image.Image) -> Image.Image:
    """
    Crops out the sky using a pretrained ADE20K segmentation model, so the
    classifier and Grad-CAM focus on the hillside itself rather than
    sky/clouds. Falls back to the original image if little/no sky is
    detected (e.g. close-up crops).

    Applied identically at training time (see crop_dataset_sky.py) and
    inference time (predict/gradcam_overlay below) — cropping only at
    inference would create a train/inference mismatch, since the model
    would suddenly see framing very different from what it was trained on.
    """
    processor, seg_model = _load_segmentation_model()
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        logits = seg_model(**inputs).logits
    upsampled = nn.functional.interpolate(
        logits, size=image.size[::-1], mode="bilinear", align_corners=False
    )
    pred = upsampled.argmax(dim=1)[0].numpy()
    sky_mask = pred == SKY_CLASS_INDEX

    if sky_mask.mean() < SKY_PIXEL_MIN_FRACTION:
        return image

    h, w = sky_mask.shape
    lowest_sky_per_col = [
        np.where(sky_mask[:, c])[0].max()
        for c in range(w) if sky_mask[:, c].any()
    ]
    if not lowest_sky_per_col:
        return image

    # Median across columns is robust to a handful of columns with stray
    # sky pixels (e.g. gaps between rocks) pulling the estimate too deep.
    crop_top = int(np.median(lowest_sky_per_col))
    crop_top = min(crop_top, int(h * SKY_CROP_MAX_FRACTION))

    return image.crop((0, crop_top, image.width, h))


def assess(unstable_prob: float) -> str:
    """Turn a raw P(unstable) into the actual (threshold-adjusted) call."""
    if unstable_prob >= UNSTABLE_HIGH_CONFIDENCE:
        return "Potentially Unstable — visible surface indicators present (e.g. cracks, scarps, loose debris, disturbed soil)."
    elif unstable_prob >= UNSTABLE_THRESHOLD:
        return "Potentially Unstable (borderline) — some possible indicators present. Field inspection recommended."
    else:
        return "Stable — no significant visible surface indicators detected."


def predict(image: Image.Image) -> dict:
    """
    Takes a PIL image, returns {"scores": {class: prob, ...}, "assessment": str}.
    """
    model, class_names, device = load_model()

    image = image.convert("RGB")
    image = crop_sky(image)
    tensor = _transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    scores = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    assessment = assess(scores.get("unstable", 0.0))

    return {"scores": scores, "assessment": assessment}


def gradcam_overlay(image: Image.Image) -> Image.Image:
    """
    Takes a PIL image, returns a PIL image with the Grad-CAM heatmap overlaid,
    showing which regions most influenced the prediction.
    """
    load_model()  # ensures _cam is initialized
    device = _device

    image = image.convert("RGB")
    image = crop_sky(image)
    rgb_img = np.array(image.resize((224, 224))).astype(np.float32) / 255.0
    input_tensor = _transform(image).unsqueeze(0).to(device)

    grayscale_cam = _cam(input_tensor=input_tensor, targets=None)[0]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    return Image.fromarray(visualization)
