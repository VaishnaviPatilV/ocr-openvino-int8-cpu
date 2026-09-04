import { FilePlus2, Loader2, ScanText } from "lucide-react";
import React from "react";

export default function ActionButtons({
  canExtract,
  isProcessing,
  onExtract,
  onNewDocument,
}) {
  return (
    <div className="action-bar" role="toolbar" aria-label="OCR actions">
      <button
        type="button"
        className="btn btn-primary btn-lg"
        onClick={onExtract}
        disabled={!canExtract || isProcessing}
        aria-busy={isProcessing}
      >
        {isProcessing ? (
          <React.Fragment>
            <Loader2 className="spin" size={18} />
            Extracting text…
          </React.Fragment>
        ) : (
          <React.Fragment>
            <ScanText size={18} />
            Extract Text
          </React.Fragment>
        )}
      </button>

      <button
        type="button"
        className="btn btn-secondary btn-lg"
        onClick={onNewDocument}
        disabled={isProcessing}
      >
        <FilePlus2 size={18} />
        New Document
      </button>
    </div>
  );
}