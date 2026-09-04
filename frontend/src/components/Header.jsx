import { ScanText } from "lucide-react";

export default function Header({ health, healthLoading }) {
  const online =
    Boolean(health) &&
    health.status === "running" &&
    health.glm_ocr_int8_loaded === true;

  return (
    <header className="app-header">
      <div className="container header-inner">
        <div className="brand">
          <div className="brand-logo" aria-hidden="true">
            <ScanText size={24} />
          </div>
          <div className="brand-text">
            <h1>OCR Model</h1>
            <p>Document &amp; Image Text Extraction</p>
          </div>
        </div>

        <div
          className={`engine-pill ${
            healthLoading ? "checking" : online ? "online" : "offline"
          }`}
          role="status"
          aria-live="polite"
        >
          <span className="pill-dot" aria-hidden="true" />
          {healthLoading
            ? "Checking engine…"
            : online
            ? "CPU OCR Engine Online"
            : "OCR Engine Offline"}
        </div>
      </div>
    </header>
  );
}