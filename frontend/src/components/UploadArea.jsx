import { Image as ImageIcon, UploadCloud } from "lucide-react";
import { useRef, useState } from "react";

export default function UploadArea({ onFileSelect, error }) {
  const inputRef = useRef(null);
  const [dragActive, setDragActive] = useState(false);

  const openFileDialog = () => inputRef.current?.click();

  const handleDragEnter = (e) => {
    e.preventDefault();
    setDragActive(true);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    if (!dragActive) setDragActive(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    if (!e.currentTarget.contains(e.relatedTarget)) {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragActive(false);
    const files = e.dataTransfer?.files;
    if (files && files.length > 0) {
      onFileSelect(files[0]); // single image only
    }
  };

  return (
    <section className="card upload-card" aria-label="Upload image">
      <div
        className={`dropzone ${dragActive ? "drag-active" : ""}`}
        role="button"
        tabIndex={0}
        aria-label="Upload your image. Drag and drop an image here, or press Enter to browse files."
        onClick={openFileDialog}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            openFileDialog();
          }
        }}
        onDragEnter={handleDragEnter}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <div className="dropzone-icon" aria-hidden="true">
          <UploadCloud size={38} strokeWidth={1.5} />
        </div>

        <h2>Upload your image</h2>
        <p className="muted">Drag &amp; drop an image here</p>

        <div className="dropzone-or" aria-hidden="true">
          <span>or</span>
        </div>

        <button
          type="button"
          className="btn btn-primary"
          onClick={(e) => {
            e.stopPropagation();
            openFileDialog();
          }}
        >
          <ImageIcon size={16} />
          Browse Files
        </button>

        <p className="supported muted">Supported: JPG, JPEG, PNG, BMP, WEBP</p>
      </div>

      {/* Hidden native file input (triggered programmatically) */}
      <input
        ref={inputRef}
        type="file"
        accept=".jpg,.jpeg,.png,.bmp,.webp,image/jpeg,image/png,image/bmp,image/webp"
        className="visually-hidden"
        aria-hidden="true"
        tabIndex={-1}
        onChange={(e) => {
          const f = e.target.files && e.target.files[0];
          if (f) onFileSelect(f);
          e.target.value = ""; // allow re-selecting the same file
        }}
      />

      {error && (
        <p className="field-error" role="alert">
          {error}
        </p>
      )}
    </section>
  );
}