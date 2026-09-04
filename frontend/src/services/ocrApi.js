const API_BASE_URL = "http://127.0.0.1:8082";

export class ApiError extends Error {
  constructor(code, message, status = null) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.status = status;
  }
}

export async function checkHealth() {
  const url = `${API_BASE_URL}/api/health`;

  console.log("[OCR UI] Checking backend:", url);

  try {
    const response = await fetch(url, {
      method: "GET",
      cache: "no-store"
    });

    console.log("[OCR UI] Health status:", response.status);

    const data = await response.json();

    console.log("[OCR UI] Health response:", data);

    if (!response.ok) {
      throw new ApiError(
        "HTTP_ERROR",
        `Health check failed: HTTP ${response.status}`,
        response.status
      );
    }

    return data;
  } catch (error) {
    console.error("[OCR UI] Health error:", error);

    if (error instanceof ApiError) {
      throw error;
    }

    throw new ApiError(
      "OFFLINE",
      "Cannot reach the OCR server."
    );
  }
}

export async function performOCR(file) {
  if (!file) {
    throw new ApiError("NO_FILE", "No file selected.");
  }

  const formData = new FormData();
  formData.append("file", file);

  console.log("[OCR UI] Sending OCR request:", file.name);

  try {
    const response = await fetch(
      `${API_BASE_URL}/api/ocr/file`,
      {
        method: "POST",
        body: formData
      }
    );

    console.log("[OCR UI] OCR HTTP status:", response.status);

    let payload = null;

    try {
      payload = await response.json();
    } catch {
      throw new ApiError(
        "BAD_RESPONSE",
        "OCR server returned invalid JSON."
      );
    }

    console.log("[OCR UI] OCR response:", payload);

    if (!response.ok) {
      throw new ApiError(
        "HTTP_ERROR",
        payload?.detail ||
          `OCR request failed: HTTP ${response.status}`,
        response.status
      );
    }

    return payload;
  } catch (error) {
    console.error("[OCR UI] OCR error:", error);

    if (error instanceof ApiError) {
      throw error;
    }

    throw new ApiError(
      "OFFLINE",
      "Cannot reach the OCR server."
    );
  }
}