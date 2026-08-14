export default function ConfidenceBars({ scores }) {
  if (!scores) return null;

  const entries = Object.entries(scores).sort((a, b) => b[1] - a[1]);

  return (
    <div className="confidence-bars">
      <h3>Raw Model Confidence</h3>
      {entries.map(([label, value]) => (
        <div className="confidence-row" key={label}>
          <div className="confidence-label">
            <span>{label}</span>
            <span>{(value * 100).toFixed(1)}%</span>
          </div>
          <div className="confidence-track">
            <div
              className={`confidence-fill confidence-fill--${label}`}
              style={{ width: `${value * 100}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}
