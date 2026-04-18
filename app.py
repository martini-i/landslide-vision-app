"""
app.py — Gradio web app for the Landslide Detection classifier.
Run with: python app.py
Then open the local URL shown in your terminal.

Install Gradio first: pip install gradio
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import gradio as gr

# ===== CONFIG =====
MODEL_PATH = "slope_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== LOAD MODEL =====
checkpoint = torch.load(MODEL_PATH, map_location=device)
class_names = checkpoint["class_names"]
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

# ===== TRANSFORM =====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ===== PREDICTION FUNCTION =====
def predict(image: Image.Image):
    """
    Takes a PIL image, returns a dict of class -> confidence score.
    Gradio Label component renders this as a bar chart automatically.
    """
    if image is None:
        return {"Error": 1.0}

    # Convert to RGB in case of RGBA/grayscale uploads
    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    scores = {class_names[i]: float(probs[i]) for i in range(len(class_names))}

    # Determine risk level for the description
    unstable_prob = scores.get("unstable", 0.0)
    if unstable_prob >= 0.75:
        risk = "HIGH RISK — Visible instability signs detected. Do not approach."
    elif unstable_prob >= 0.45:
        risk = "MODERATE RISK — Some warning signs present. Exercise caution."
    else:
        risk = "LOW RISK — No significant instability signs detected."

    return scores, risk

# ===== GRADIO UI =====
with gr.Blocks(title="Landslide Detection App", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🏔️ Landslide Detection App
    Upload a **ground-level photo of a slope** to detect visible instability warning signs.

    **Detects:** tension cracks · fresh scarps · loose debris · exposed/disturbed soil

    > ⚠️ This tool detects *visible surface features only*. It does not predict future landslides.
    > Always consult a geotechnical engineer for safety decisions.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Slope Image")
            submit_btn  = gr.Button("Analyze", variant="primary")

        with gr.Column(scale=1):
            label_output = gr.Label(num_top_classes=2, label="Classification Confidence")
            risk_output  = gr.Textbox(label="Risk Assessment", lines=2)

    submit_btn.click(
        fn=predict,
        inputs=image_input,
        outputs=[label_output, risk_output]
    )

    gr.Examples(
        examples=[
            ["slope_dataset/train/stable/stable_cliff_001.jpg"],
            ["slope_dataset/train/unstable/unstable_crack_001.jpg"],
        ],
        inputs=image_input,
        label="Try an example"
    )

    gr.Markdown("""
    ---
    **How to interpret results:**
    - `stable` confidence > 70% → slope shows no major surface warning signs
    - `unstable` confidence > 70% → visible signs of potential instability detected
    - Scores between 45–70% → borderline; field inspection recommended
    """)

if __name__ == "__main__":
    demo.launch(share=False)  # set share=True to get a public link
