# syntax=docker/dockerfile:1

# ---- Stage 1: build the React frontend into static files ----
FROM node:20-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: Python backend serving the model + the built frontend ----
FROM python:3.11-slim
WORKDIR /app

# System deps for opencv (pulled in by grad-cam) and Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt backend/requirements.txt
# CPU-only torch build — much smaller than the default CUDA wheels, and this is
# intended for a homelab NAS without GPU passthrough. Drop --index-url if you do
# have a GPU passed through to the container and want CUDA acceleration.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision \
    && pip install --no-cache-dir -r backend/requirements.txt

COPY model_utils.py .
COPY slope_model.pth .
COPY backend/ backend/
COPY --from=frontend-build /app/frontend/dist frontend/dist

EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
