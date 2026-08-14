export default function AssessmentCard({ assessment }) {
  if (!assessment) return null;

  const isUnstable = assessment.startsWith("Potentially Unstable");
  const tone = isUnstable ? "unstable" : "stable";

  return (
    <div className={`assessment-card assessment-card--${tone}`}>
      <h3>Assessment (this is the actual call)</h3>
      <p>{assessment}</p>
    </div>
  );
}
