import { AlertTriangle, Loader2, ScanText } from "lucide-react";
import React, { useCallback, useEffect, useRef, useState } from "react";

import ActionButtons from "./components/ActionButtons.jsx";
import Header from "./components/Header.jsx";
import HealthStatus from "./components/HealthStatus.jsx";
import ImagePreview from "./components/ImagePreview.jsx";
import OCRResult from "./components/OCRResult.jsx";
import UploadArea from "./components/UploadArea.jsx";
import { checkHealth, performOCR } from "./services/ocrApi.js";

// ----------------------------------------------------------
// Configuration
// ----------------------------------------------------------
const ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png", "bmp", "webp"];
const ACCEPT_ATTR = ".jpg,.jpeg,.png,.bmp,.webp";

const MSG_UNSUPPORTED =
  "Unsupported file type. Please upload JPG, JPEG, PNG, BMP, or WEBP.";
const MSG_INVALID_IMAGE =
  "The uploaded file could not be read. Please select a valid image.";
const MSG_BACKEND_OFFLINE =
  "OCR server is unavailable. Make sure the FastAPI server is running at http://127.0.0.1:8082";
const MSG_OCR_FAILED = "OCR failed. Please try another image.";

function formatFileSize(bytes) {
  if (!Number.isFinite(bytes)) return "";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

// ----------------------------------------------------------
// Right-panel states
// ----------------------------------------------------------
function ProcessingPanel() {
  return (
    <section className="card panel-center" role="status" aria-live="polite">
      <Loader2 className="spin" size={40} strokeWidth={1.75} />
      <h2>Analyzing document…</h2>
      <p className="muted">GLM-OCR is extracting text</p>
    </section>
  );
}

function PlaceholderPanel() {
  return (
    <section className="card panel-center placeholder">
      <ScanText size={44} strokeWidth={1.25} />
      <h2>Ready to extract</h2>
      <p className="muted">
        Click “Extract Text” to run OCR on this image. The extracted text will
        appear here.
      </p>
    </section>
  );
}

export default function App() {
  // ---- Engine health ----
  const [health, setHealth] = useState(null);
  const [healthLoading, setHealthLoading] = useState(true);

  // ---- Selected file ----
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [imageMeta, setImageMeta] = useState(null);
  const [fileError, setFileError] = useState(null);

  // ---- OCR run ----
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [ocrError, setOcrError] = useState(null);

  const changeInputRef = useRef(null);
  const previewUrlRef = useRef(null);

  // ----------------------------------------------------------
  // Health check (on mount)
  // ----------------------------------------------------------
  const refreshHealth = useCallback(async () => {
    setHealthLoading(true);
    try {
      const data = await checkHealth();
      setHealth(data);
    } catch {
      setHealth(null);
    } finally {
      setHealthLoading(false);
    }
  }, []);

  useEffect(() => {
    refreshHealth();
  }, [refreshHealth]);

  // ----------------------------------------------------------
  // Object-URL lifecycle (avoid memory leaks)
  // ----------------------------------------------------------
  useEffect(() => {
    return () => {
      if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    };
  }, []);

  const clearPreview = useCallback(() => {
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current);
      previewUrlRef.current = null;
    }
    setPreviewUrl(null);
    setImageMeta(null);
  }, []);

  // ----------------------------------------------------------
  // File selection (extension check + preview + dimensions)
  // ----------------------------------------------------------
  const handleFileSelect = useCallback(
    (selected) => {
      setFileError(null);
      setOcrError(null);
      setResult(null);

      const ext = (selected?.name || "").split(".").pop().toLowerCase();
      if (!ALLOWED_EXTENSIONS.includes(ext)) {
        setFileError(MSG_UNSUPPORTED);
        return;
      }

      clearPreview();

      const url = URL.createObjectURL(selected);
      previewUrlRef.current = url;
      setFile(selected);
      setPreviewUrl(url);

      // Read dimensions for the metadata line
      const img = new Image();
      img.onload = () => {
        setImageMeta({
          width: img.naturalWidth,
          height: img.naturalHeight,
          size: formatFileSize(selected.size),
        });
      };
      img.onerror = () => {
        // Corrupted / unreadable image → reject it
        clearPreview();
        setFile(null);
        setFileError(MSG_INVALID_IMAGE);
      };
      img.src = url;
    },
    [clearPreview]
  );

  // Reset everything back to the upload state
  const resetSelection = useCallback(() => {
    clearPreview();
    setFile(null);
    setFileError(null);
    setResult(null);
    setOcrError(null);
  }, [clearPreview]);

  const handleChangeFile = useCallback(() => {
    changeInputRef.current?.click();
  }, []);

  // ----------------------------------------------------------
  // OCR request
  // ----------------------------------------------------------
  const handleExtractText = async () => {
    if (!file || isProcessing) return; // guard against duplicate requests

    setIsProcessing(true);
    setOcrError(null);
    setResult(null);

    try {
      const res = await performOCR(file);

      if (res && res.error) {
        // Backend returned success:false with an error field
        setOcrError(MSG_OCR_FAILED);
      } else if (res) {
        // success:true OR empty text — both render in OCRResult
        setResult(res);
      } else {
        setOcrError(MSG_OCR_FAILED);
      }
    } catch (err) {
      if (err?.code === "OFFLINE") {
        setOcrError(MSG_BACKEND_OFFLINE);
      } else if (err?.status === 400) {
        setOcrError(MSG_INVALID_IMAGE);
      } else {
        setOcrError(err?.message || MSG_OCR_FAILED);
      }
    } finally {
      setIsProcessing(false);
    }
  };

  // ----------------------------------------------------------
  // Copy text (Clipboard API + fallback for non-secure contexts)
  // ----------------------------------------------------------
  const handleCopyText = async () => {
    if (!result?.text) return false;
    try {
      await navigator.clipboard.writeText(result.text);
      return true;
    } catch {
      try {
        const ta = document.createElement("textarea");
        ta.value = result.text;
        ta.setAttribute("readonly", "");
        ta.style.position = "fixed";
        ta.style.opacity = "0";
        document.body.appendChild(ta);
        ta.select();
        const ok = document.execCommand("copy");
        document.body.removeChild(ta);
        return ok;
      } catch {
        return false;
      }
    }
  };

  // ----------------------------------------------------------
  // Download TXT (client-side blob, original filename → .txt)
  // ----------------------------------------------------------
  const handleDownloadTxt = () => {
    if (!result?.text) return;

    const base = file?.name || result.filename || "ocr_result";
    const txtName = base.replace(/\.[^.]+$/, "") + ".txt";

    const blob = new Blob([result.text], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = txtName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);

    URL.revokeObjectURL(url);
  };

  // ----------------------------------------------------------
  // Render
  // ----------------------------------------------------------
  const visibleError = fileError || ocrError;

  return (
    <div className="app">
      <Header health={health} healthLoading={healthLoading} />

      <main className="container main-content">
        <div className="layout">
          {/* ---------- Primary column ---------- */}
          <div className="primary-col">
            {!file ? (
              <UploadArea onFileSelect={handleFileSelect} error={fileError} />
            ) : (
              <React.Fragment>
                <ActionButtons
                  isProcessing={isProcessing}
                  canExtract={Boolean(file)}
                  onExtract={handleExtractText}
                  onNewDocument={resetSelection}
                />

                {visibleError && (
                  <div className="banner banner-error" role="alert">
                    <AlertTriangle size={18} />
                    <span>{visibleError}</span>
                  </div>
                )}

                <div className="workspace-grid">
                  <ImagePreview
                    file={file}
                    previewUrl={previewUrl}
                    meta={imageMeta}
                    onRemove={resetSelection}
                    onChangeFile={handleChangeFile}
                    isProcessing={isProcessing}
                  />

                  <div className="result-slot">
                    {isProcessing ? (
                      <ProcessingPanel />
                    ) : result ? (
                      <OCRResult
                        result={result}
                        onCopy={handleCopyText}
                        onDownload={handleDownloadTxt}
                      />
                    ) : (
                      <PlaceholderPanel />
                    )}
                  </div>
                </div>
              </React.Fragment>
            )}
          </div>

          {/* ---------- Sidebar ---------- */}
          <aside className="secondary-col">
            <HealthStatus
              health={health}
              loading={healthLoading}
              onRefresh={refreshHealth}
            />
          </aside>
        </div>
      </main>

      <footer className="app-footer">
        <p className="muted">OCR Model· OpenVINO INT8 · CPU-only inference</p>
      </footer>

      {/* Hidden input used by "Change File" */}
      <input
        ref={changeInputRef}
        type="file"
        accept={ACCEPT_ATTR}
        className="visually-hidden"
        aria-hidden="true"
        tabIndex={-1}
        onChange={(e) => {
          const f = e.target.files && e.target.files[0];
          if (f) handleFileSelect(f);
          e.target.value = "";
        }}
      />
    </div>
  );
}