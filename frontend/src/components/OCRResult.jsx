import {
    Check,
    Clock,
    Copy,
    Download,
    File as FileIcon,
    FileSearch,
    FileText,
} from "lucide-react";
import { useState } from "react";

function toTxtName(filename) {
  const base = filename || "ocr_result";
  return base.replace(/\.[^.]+$/, "") + ".txt";
}

export default function OCRResult({ result, onCopy, onDownload }) {
  const [copied, setCopied] = useState(false);

  const handleCopyClick = async () => {
    const ok = await onCopy();
    if (ok) {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    }
  };

  const hasText = Boolean(result?.text);
  const time = Number(result?.processing_time ?? 0).toFixed(2);

  return (
    <section className="card result-card" aria-label="OCR result">
      <div className="card-header">
        <h2>
          <FileText size={17} /> OCR Result
        </h2>
      </div>

      <dl className="result-stats">
        <div>
          <dt>
            <FileIcon size={13} /> File
          </dt>
          <dd title={result.filename}>{result.filename}</dd>
        </div>
        <div>
          <dt>
            <Clock size={13} /> Processing Time
          </dt>
          <dd>{time} seconds</dd>
        </div>
        <div>
          <dt>
            <FileText size={13} /> Text File
          </dt>
          <dd title={result.text_file || toTxtName(result.filename)}>
            {toTxtName(result.filename)}
          </dd>
        </div>
      </dl>

      {hasText ? (
        <textarea
          className="ocr-text"
          readOnly
          value={result.text}
          wrap="off"
          spellCheck={false}
          aria-label="Extracted OCR text"
        />
      ) : (
        <div className="empty-notice" role="status">
          <FileSearch size={18} />
          No readable text was detected in this image.
        </div>
      )}

      <div className="result-actions">
        <button
          type="button"
          className="btn btn-secondary"
          onClick={handleCopyClick}
          disabled={!hasText}
        >
          {copied ? <Check size={16} /> : <Copy size={16} />}
          {copied ? "Copied!" : "Copy Text"}
        </button>

        <button
          type="button"
          className="btn btn-primary"
          onClick={onDownload}
          disabled={!hasText}
        >
          <Download size={16} /> Download TXT
        </button>
      </div>

      {result.text_file && (
        <p className="server-path muted" title={result.text_file}>
          Saved on server: {result.text_file}
        </p>
      )}
    </section>
  );
}