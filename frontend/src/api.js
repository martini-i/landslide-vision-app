// In dev, the frontend (Vite, :5173) and backend (uvicorn, :8000) are separate
// servers, so requests need the full backend URL. In production the built frontend
// is served by the same FastAPI process, so relative paths hit the right place
// automatically regardless of host/port.
const API_BASE = import.meta.env.DEV ? "http://localhost:8000" : "";

export async function predictImage(file) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    throw new Error(`Prediction failed (${res.status})`);
  }

  return res.json(); // { scores: { stable, unstable }, assessment }
}

export async function gradcamImage(file) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE}/gradcam`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    throw new Error(`Grad-CAM failed (${res.status})`);
  }

  const blob = await res.blob();
  return URL.createObjectURL(blob);
}
