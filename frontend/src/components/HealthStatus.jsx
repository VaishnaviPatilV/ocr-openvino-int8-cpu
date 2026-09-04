import { Cpu, RefreshCw, Server } from "lucide-react";

export default function HealthStatus({
  health,
  loading,
  onRefresh
}) {
  const online =
    Boolean(health) &&
    health.status === "running" &&
    health.ocrmodel_int8_loaded === true;

  const devices =
    health?.openvino_devices?.length
      ? health.openvino_devices.join(", ")
      : "—";

  return (
    <section className="card health-card">
      <div className="card-header">
        <div>
          <h2>OCR Engine</h2>
          <p className="muted">
            Runtime information
          </p>
        </div>

        <button
          type="button"
          className="icon-button"
          onClick={onRefresh}
          disabled={loading}
          title="Refresh engine status"
        >
          <RefreshCw
            size={16}
            className={loading ? "spin" : ""}
          />
        </button>
      </div>

      <div className="health-status">
        <span
          className={`status-dot ${
            online ? "status-online" : "status-offline"
          }`}
        />

        <strong>
          {loading
            ? "Checking..."
            : online
            ? "Online"
            : "Offline"}
        </strong>
      </div>

      <dl className="health-list">
        <div className="health-row">
          <dt>Model</dt>
          <dd>OCR Model</dd>
        </div>

        <div className="health-row">
          <dt>Runtime</dt>
          <dd>OpenVINO</dd>
        </div>

        <div className="health-row">
          <dt>Device</dt>
          <dd>
            <Cpu size={14} />
            {health?.device || "CPU"}
          </dd>
        </div>

        <div className="health-row">
          <dt>OpenVINO devices</dt>
          <dd>
            <Server size={14} />
            {devices}
          </dd>
        </div>

        {health?.max_new_tokens != null && (
          <div className="health-row">
            <dt>Max tokens</dt>
            <dd>{health.max_new_tokens}</dd>
          </div>
        )}

        {health?.max_concurrent_inference != null && (
          <div className="health-row">
            <dt>Concurrency</dt>
            <dd>{health.max_concurrent_inference}</dd>
          </div>
        )}
      </dl>

      {!online && !loading && (
        <div className="health-hint">
          <strong>Backend connection issue</strong>
          <p>
            Make sure FastAPI is running at:
          </p>
          <code>http://127.0.0.1:8082</code>
        </div>
      )}
    </section>
  );
}