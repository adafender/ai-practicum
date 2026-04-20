export default function ReportScreen({ report }) {
  return (
    <div>
      <h1>Report Card</h1>
      <pre style={{ whiteSpace: "pre-wrap" }}>
        {report}
      </pre>
    </div>
  );
}