import { useState, useEffect } from "react";
import ImageUpload from "./components/ImageUpload";
import ConfidenceBars from "./components/ConfidenceBars";
import AssessmentCard from "./components/AssessmentCard";
import { predictImage, gradcamImage } from "./api";
import "./App.css";

const EXAMPLES = [
  { label: "Stable — natural rock", tone: "stable", path: "/examples/stable_cliff_001.jpg" },
  { label: "Stable — engineered", tone: "stable", path: "/examples/stable_engineered_001.jpg" },
  { label: "Unstable — crack", tone: "unstable", path: "/examples/unstable_crack_001.jpg" },
  { label: "Unstable — scarp", tone: "unstable", path: "/examples/unstable_scarp_007.jpg" },
];

function Spinner() {
  return <span className="spinner" aria-hidden="true" />;
}

export default function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [gradcamUrl, setGradcamUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [theme, setTheme] = useState(() => localStorage.getItem("theme") || "dark");

  useEffect(() => {
    localStorage.setItem("theme", theme);
  }, [theme]);

  function handleFileSelected(selectedFile) {
    setFile(selectedFile);
    setPreviewUrl(URL.createObjectURL(selectedFile));
    setResult(null);
    setGradcamUrl(null);
    setError(null);
  }

  async function handleExampleClick(path) {
    const res = await fetch(path);
    const blob = await res.blob();
    const exampleFile = new File([blob], path.split("/").pop(), { type: blob.type });
    handleFileSelected(exampleFile);
  }

  async function handleAnalyze() {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const prediction = await predictImage(file);
      setResult(prediction);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleShowGradcam() {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const url = await gradcamImage(file);
      setGradcamUrl(url);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app" data-theme={theme}>
      <header>
        <div className="top-bar">
          <div className="brand">
            <span className="brand-mark" aria-hidden="true">
              <svg width="26" height="26" viewBox="0 0 24 24" fill="none">
                <path d="M2 19h20L15 6l-4 7-3-4-6 10z" fill="currentColor" opacity="0.9" />
              </svg>
            </span>
            <div>
              <h1>GroundTruth</h1>
              <p className="subtitle">Slope Surface Indicator Classifier</p>
            </div>
          </div>
          <button
            className="theme-toggle"
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          >
            {theme === "dark" ? "☀ Light mode" : "☾ Dark mode"}
          </button>
        </div>
        <p>
          Upload a <strong>ground-level photo of a slope</strong> to check for visible
          surface indicators associated with potential slope instability.
        </p>
        <p className="looks-for">
          <strong>Looks for:</strong> tension cracks · fresh scarps · loose debris/talus ·
          exposed or disturbed soil · undercutting · rockfall evidence
        </p>
        <blockquote>
          This tool identifies visible surface indicators associated with potential slope
          instability. It does <strong>not</strong> predict landslides, determine whether a
          landslide will occur, or assess subsurface geotechnical conditions. It is a research
          prototype, not a safety determination — always consult a qualified geotechnical
          professional for safety decisions.
        </blockquote>
      </header>

      <main className="main-grid">
        <section className="upload-section">
          <ImageUpload onFileSelected={handleFileSelected} previewUrl={previewUrl} />

          <div className="button-row">
            <button onClick={handleAnalyze} disabled={!file || loading} className="btn-primary">
              {loading ? <Spinner /> : null}
              {loading ? "Working…" : "Analyze"}
            </button>
            <button onClick={handleShowGradcam} disabled={!file || loading} className="btn-secondary">
              Show Grad-CAM
            </button>
          </div>

          <div className="examples">
            <span className="examples-label">Try an example</span>
            <div className="examples-grid">
              {EXAMPLES.map((ex) => (
                <button
                  key={ex.path}
                  className="example-card"
                  onClick={() => handleExampleClick(ex.path)}
                  title={ex.label}
                >
                  <img src={ex.path} alt={ex.label} />
                  <span className={`example-tag example-tag--${ex.tone}`}>{ex.label}</span>
                </button>
              ))}
            </div>
          </div>
        </section>

        <section className="results-section">
          {error && <p className="error">{error}</p>}
          {!result && !gradcamUrl && !error && (
            <div className="placeholder">
              <p>Results will appear here once you analyze an image.</p>
            </div>
          )}
          {result && (
            <>
              <ConfidenceBars scores={result.scores} />
              <AssessmentCard assessment={result.assessment} />
            </>
          )}
          {gradcamUrl && (
            <div className="gradcam-panel">
              <h3>Grad-CAM — where the model is looking</h3>
              <img src={gradcamUrl} alt="Grad-CAM heatmap" className="gradcam-image" />
            </div>
          )}
        </section>
      </main>

      <footer>
        <p>
          <strong>How to interpret results:</strong> the Assessment card is the actual call — it
          uses a 35% threshold on P(unstable), not 50%, because missing a genuinely unstable slope
          is treated as a costlier error than a false alarm. It can disagree with which class the
          raw confidence bars show as highest — that's intentional, not a bug.
        </p>
        <ul>
          <li>P(unstable) ≥ 65% → Potentially Unstable</li>
          <li>35% ≤ P(unstable) &lt; 65% → Potentially Unstable (borderline) — inspect further</li>
          <li>P(unstable) &lt; 35% → Stable</li>
        </ul>
      </footer>
    </div>
  );
}
