export default function DatasetProfile({ profile }) {
  if (!profile) return null;

  return (
    <div className="panel">
      <h2>2. Dataset profile</h2>
      <p style={{ fontSize: "0.85rem", color: "#9aa2ad" }}>
        {profile.n_rows} rows × {profile.n_columns} columns
      </p>
      <table style={{ width: "100%", fontSize: "0.85rem", borderCollapse: "collapse" }}>
        <thead>
          <tr style={{ textAlign: "left", color: "#9aa2ad" }}>
            <th>Column</th>
            <th>Kind</th>
            <th>Missing</th>
            <th>Unique</th>
          </tr>
        </thead>
        <tbody>
          {profile.columns.map((col) => (
            <tr key={col.name} style={{ borderTop: "1px solid #2a2e37" }}>
              <td>{col.name}</td>
              <td>{col.kind}</td>
              <td>{(col.missing_fraction * 100).toFixed(1)}%</td>
              <td>{col.n_unique}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
