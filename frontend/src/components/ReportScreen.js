export default function ReportScreen({ report }) {
  // Split report into sections (basic parsing)
  const sections = report ? report.split("\n") : [];

  return (
    <div className="card">
      <h2>📊 Performance Report Card</h2>

      <div className="report-container">
        {sections.map((line, index) => {
          // Highlight score lines
          if (line.toLowerCase().includes("score")) {
            return (
              <div key={index} className="report-score">
                {line}
              </div>
            );
          }

          // Highlight strengths
          if (line.toLowerCase().includes("strength")) {
            return (
              <div key={index} className="report-strength">
                {line}
              </div>
            );
          }

          // Highlight improvements
          if (line.toLowerCase().includes("improvement")) {
            return (
              <div key={index} className="report-improvement">
                {line}
              </div>
            );
          }

          return <p key={index}>{line}</p>;
        })}
      </div>
    </div>
  );
}