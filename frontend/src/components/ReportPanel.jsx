import { useState } from "react";

export default function ReportPanel({ datasetId, columns }) {
  const [target, setTarget] = useState("");
  const [includeHypotheses, setIncludeHypotheses] = useState(false);
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(false);

  async function runReport() {
    setLoading(true);
    try {
      const res = await fetch("/api/report", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_id: datasetId,
          target_column: target || null,
          include_hypotheses: includeHypotheses,
        }),
      });
      const data = await res.json();
      setReport(data);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="panel">
      <h2>4. Report &amp; hypotheses</h2>
      <div style={{ marginBottom: "0.75rem" }}>
        <select value={target} onChange={(e) => setTarget(e.target.value)}>
          <option value="">No target column (unsupervised only)</option>
          {columns
            .filter((c) => c.kind !== "identifier")
            .map((c) => (
              <option key={c.name} value={c.name}>
                {c.name}
              </option>
            ))}
        </select>
        <label style={{ marginLeft: "1rem", fontSize: "0.85rem" }}>
          <input
            type="checkbox"
            checked={includeHypotheses}
            onChange={(e) => setIncludeHypotheses(e.target.checked)}
          />{" "}
          Include AI-generated hypotheses
        </label>
      </div>
      <button onClick={runReport} disabled={loading}>
        {loading ? "Generating…" : "Generate report"}
      </button>

      {report && (
        <div style={{ marginTop: "1rem" }}>
          <pre>{report.markdown}</pre>
          {report.hypotheses?.length > 0 && (
            <div>
              <h3 style={{ fontSize: "0.95rem" }}>Candidate hypotheses</h3>
              {report.hypotheses.map((h, i) => (
                <div className="hypothesis-card" key={i}>
                  <span className="hypothesis-label">{h.label}</span>
                  {h.text}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
