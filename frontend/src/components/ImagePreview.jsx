import { FileImage, Loader2, RefreshCw, X } from "lucide-react";

export default function ImagePreview({
  file,
  previewUrl,
  meta,
  onRemove,
  onChangeFile,
  isProcessing,
}) {
  const metaParts = [];
  if (meta?.width && meta?.height) {
    metaParts.push(`${meta.width} × ${meta.height} px`);
  }
  if (meta?.size) metaParts.push(meta.size);

  return (
    <section className="card preview-card" aria-label="Selected file">
      <div className="card-header">
        <h2>
          <FileImage size={17} /> Selected File
        </h2>
      </div>

      <div className="filename-chip" title={file.name}>
        {file.name}
      </div>

      <div className="preview-stage">
        {previewUrl && <img src={previewUrl} alt={`Preview of ${file.name}`} />}

        {isProcessing && (
          <div className="processing-overlay" role="status" aria-live="polite">
            <Loader2 className="spin" size={34} />
            <p className="overlay-title">Analyzing document…</p>
            <p className="muted">OCR Model is extracting text</p>
          </div>
        )}
      </div>

      <p className="preview-meta muted">
        {metaParts.length > 0 ? metaParts.join("   ·   ") : "\u00A0"}
      </p>

      <div className="preview-actions">
        <button
          type="button"
          className="btn btn-secondary btn-sm"
          onClick={onRemove}
        >
          <X size={15} /> Remove
        </button>
        <button
          type="button"
          className="btn btn-secondary btn-sm"
          onClick={onChangeFile}
        >
          <RefreshCw size={15} /> Change File
        </button>
      </div>
    </section>
  );
}