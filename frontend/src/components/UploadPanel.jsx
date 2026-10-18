import { useState } from "react";

export default function UploadPanel({ onUploaded }) {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function handleUpload() {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await fetch("/api/datasets/upload", {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `Upload failed (${res.status})`);
      }
      const data = await res.json();
      onUploaded(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="panel">
      <h2>1. Upload dataset</h2>
      <p style={{ fontSize: "0.85rem", color: "#9aa2ad" }}>
        CSV or UEA .ts time-series files. The backend auto-detects numeric, categorical, and
        time-series columns — no manual schema needed.
      </p>
      <input
        type="file"
        accept=".csv,.ts"
        onChange={(e) => setFile(e.target.files[0])}
      />
      <div style={{ marginTop: "0.75rem" }}>
        <button onClick={handleUpload} disabled={!file || loading}>
          {loading ? "Uploading…" : "Upload & profile"}
        </button>
      </div>
      {error && <p style={{ color: "#e08080" }}>{error}</p>}
    </div>
  );
}
