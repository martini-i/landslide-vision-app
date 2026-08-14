const API_BASE = "http://localhost:8000";

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
