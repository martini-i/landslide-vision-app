"""
backend/main.py — FastAPI backend for GroundTruth, wrapping model_utils.py for the React frontend.
Run from the project root with: uvicorn backend.main:app --reload --port 8000
"""

import io
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model_utils

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

app = FastAPI(title="GroundTruth API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_upload_image(file: UploadFile) -> Image.Image:
    try:
        return Image.open(io.BytesIO(file.file.read()))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    image = _load_upload_image(file)
    return model_utils.predict(image)


@app.post("/gradcam")
def gradcam(file: UploadFile = File(...)):
    image = _load_upload_image(file)
    overlay = model_utils.gradcam_overlay(image)

    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


# In production (Docker), the React app is pre-built into frontend/dist and served
# from this same FastAPI process — one container, one port. This mount must stay
# last so it doesn't shadow the API routes above; it only activates if the built
# frontend is actually present (i.e. not during local `uvicorn --reload` dev use).
_frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="frontend")
