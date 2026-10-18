import { useState } from "react";
import UploadPanel from "./components/UploadPanel.jsx";
import DatasetProfile from "./components/DatasetProfile.jsx";
import EmbeddingPlot from "./components/EmbeddingPlot.jsx";
import ReportPanel from "./components/ReportPanel.jsx";

export default function App() {
  const [profile, setProfile] = useState(null);

  return (
    <div className="app-shell">
      <h1>SciDiscover</h1>
      <p style={{ color: "#9aa2ad" }}>
        Upload a dataset. Find structure, drivers, and testable hypotheses.
      </p>

      <UploadPanel onUploaded={setProfile} />
      <DatasetProfile profile={profile} />

      {profile && (
        <>
          <EmbeddingPlot datasetId={profile.dataset_id} />
          <ReportPanel datasetId={profile.dataset_id} columns={profile.columns} />
        </>
      )}
    </div>
  );
}
