"""
app.py — Gradio interface for the slope surface-indicator classifier.
Run with: python app.py
Then open the local URL shown in your terminal.

Install Gradio first: pip install gradio
"""

from PIL import Image
import gradio as gr
import model_utils

# ===== PREDICTION FUNCTION =====
def predict(image: Image.Image):
    """Takes a PIL image, returns (scores dict, assessment string) for the Gradio outputs."""
    if image is None:
        return {"Error": 1.0}, ""

    result = model_utils.predict(image)
    return result["scores"], result["assessment"]

# ===== GRADIO UI =====
with gr.Blocks(title="Slope Surface Indicator Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Slope Surface Indicator Classifier
    Upload a **ground-level photo of a slope** to check for visible surface indicators
    associated with potential slope instability.

    **Looks for:** tension cracks · fresh scarps · loose debris/talus · exposed or disturbed soil ·
    undercutting · rockfall evidence

    > This tool identifies visible surface indicators associated with potential slope instability.
    > It does **not** predict landslides, determine whether a landslide will occur, or assess
    > subsurface geotechnical conditions. It is a research prototype, not a safety determination —
    > always consult a qualified geotechnical professional for safety decisions.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Slope Image")
            submit_btn  = gr.Button("Analyze", variant="primary")

        with gr.Column(scale=1):
            label_output = gr.Label(num_top_classes=2, label="Raw Model Confidence")
            assessment_output = gr.Textbox(label="Assessment (this is the actual call)", lines=2)

    submit_btn.click(
        fn=predict,
        inputs=image_input,
        outputs=[label_output, assessment_output]
    )

    gr.Examples(
        examples=[
            ["slope_dataset/train/stable/stable_cliff_001.jpg"],
            ["slope_dataset/train/stable/stable_engineered_001.jpg"],
            ["slope_dataset/train/unstable/unstable_crack_001.jpg"],
            ["slope_dataset/train/unstable/unstable_scarp_007.jpg"],
        ],
        inputs=image_input,
        label="Try an example"
    )

    gr.Markdown("""
    ---
    **How to interpret results:**
    - The **Assessment** box is the actual call — it uses a 35% threshold on P(unstable),
      not 50%, because missing a genuinely unstable slope is treated as a costlier error
      than a false alarm. It can disagree with which class the raw confidence bars show
      as highest — that's intentional, not a bug.
    - P(unstable) ≥ 65% → Potentially Unstable
    - 35% ≤ P(unstable) < 65% → Potentially Unstable (borderline) — inspect further
    - P(unstable) < 35% → Stable
    """)

if __name__ == "__main__":
    demo.launch(share=False)  # set share=True to get a public link
