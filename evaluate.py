"""
evaluate.py — Run after training to get full metrics on the test set.
Outputs: accuracy, precision, recall, F1, and a confusion matrix plot.
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import numpy as np
import os

# ===== CONFIG =====
DATA_DIR   = "slope_dataset"
MODEL_PATH = "slope_model.pth"
BATCH_SIZE = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== LOAD CHECKPOINT =====
checkpoint = torch.load(MODEL_PATH, map_location=device)
class_names = checkpoint["class_names"]
print(f"Classes: {class_names}")

# ===== REBUILD MODEL =====
# Detect architecture from checkpoint
arch = checkpoint.get("model_arch", "ResNet")
if "EfficientNet" in arch:
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)
else:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

# ===== TEST DATA =====
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=test_transform)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

if len(test_dataset) == 0:
    print("Test set is empty. Add images to slope_dataset/test/stable and /unstable first.")
    exit()

# ===== INFERENCE =====
all_preds, all_labels, all_probs = [], [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = probs.argmax(dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.tolist())
        all_probs.extend(probs[:, 1].cpu().tolist())  # prob of "unstable" class

# ===== METRICS =====
print("\n" + "="*50)
print("CLASSIFICATION REPORT")
print("="*50)
print(classification_report(all_labels, all_preds, target_names=class_names))

# Overall accuracy
accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
print(f"Overall Accuracy: {accuracy:.4f}")

# ROC-AUC (only meaningful if both classes are present)
if len(set(all_labels)) == 2:
    auc = roc_auc_score(all_labels, all_probs)
    print(f"ROC-AUC Score:    {auc:.4f}")

# ===== CONFUSION MATRIX PLOT =====
cm = confusion_matrix(all_labels, all_preds)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
axes[0].set_title("Confusion Matrix")

# ROC Curve
if len(set(all_labels)) == 2:
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    axes[1].plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {auc:.2f}")
    axes[1].plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve")
    axes[1].legend(loc="lower right")
else:
    axes[1].text(0.5, 0.5, "ROC requires both classes\nin test set",
                 ha="center", va="center", transform=axes[1].transAxes)

plt.tight_layout()
plt.savefig("evaluation_results.png", dpi=150)
plt.show()
print("\nPlot saved to evaluation_results.png")
