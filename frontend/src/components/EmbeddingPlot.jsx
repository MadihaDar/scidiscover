import { useState } from "react";
import Plot from "react-plotly.js";

export default function EmbeddingPlot({ datasetId }) {
  const [method, setMethod] = useState("pca");
  const [coords, setCoords] = useState(null);
  const [variance, setVariance] = useState(null);
  const [loading, setLoading] = useState(false);

  async function runEmbedding() {
    setLoading(true);
    try {
      const res = await fetch("/api/analyze/embedding", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset_id: datasetId, method, n_components: 2 }),
      });
      const data = await res.json();
      setCoords(data.coordinates);
      setVariance(data.explained_variance_ratio);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="panel">
      <h2>3. Pattern discovery</h2>
      <select value={method} onChange={(e) => setMethod(e.target.value)}>
        <option value="pca">PCA</option>
        <option value="umap">UMAP</option>
      </select>{" "}
      <button onClick={runEmbedding} disabled={loading}>
        {loading ? "Computing…" : "Run embedding"}
      </button>

      {variance && (
        <p style={{ fontSize: "0.8rem", color: "#9aa2ad" }}>
          Explained variance: {variance.map((v) => `${(v * 100).toFixed(1)}%`).join(", ")}
        </p>
      )}

      {coords && (
        <Plot
          data={[
            {
              x: coords.map((c) => c[0]),
              y: coords.map((c) => c[1]),
              mode: "markers",
              type: "scatter",
              marker: { color: "#3a7bd5", size: 8 },
            },
          ]}
          layout={{
            width: 600,
            height: 400,
            paper_bgcolor: "transparent",
            plot_bgcolor: "transparent",
            font: { color: "#e6e6e6" },
            margin: { t: 20 },
          }}
        />
      )}
    </div>
  );
}
