const rawBaseUrl = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";
export const API_BASE_URL = rawBaseUrl.replace(/\/+$/, "");

export class ApiError extends Error {
  constructor(message, status = 0) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function parseErrorMessage(response) {
  try {
    const data = await response.json();
    if (data && typeof data.detail === "string") {
      return data.detail;
    }
    if (data && Array.isArray(data.detail)) {
      return data.detail.map((item) => item.msg || JSON.stringify(item)).join("; ");
    }
  } catch {
    // Ignore JSON parse errors on non-JSON error responses
  }
  return `Server responded with HTTP ${response.status}`;
}

/**
 * Check backend health and YOLO11 model readiness.
 * @param {AbortSignal} [signal]
 */
export async function fetchHealth(signal) {
  let response;
  try {
    response = await fetch(`${API_BASE_URL}/health`, {
      method: "GET",
      headers: { Accept: "application/json" },
      signal,
    });
  } catch (err) {
    if (err.name === "AbortError") throw err;
    throw new ApiError(
      `Cannot reach backend at ${API_BASE_URL}. Ensure the FastAPI server is running.`,
      0
    );
  }

  if (!response.ok) {
    const detail = await parseErrorMessage(response);
    throw new ApiError(detail, response.status);
  }

  return response.json();
}

/**
 * Fetch supported COCO object classes from the backend.
 * @param {AbortSignal} [signal]
 */
export async function fetchClasses(signal) {
  let response;
  try {
    response = await fetch(`${API_BASE_URL}/classes`, {
      method: "GET",
      headers: { Accept: "application/json" },
      signal,
    });
  } catch (err) {
    if (err.name === "AbortError") throw err;
    throw new ApiError(`Failed to load COCO class list from ${API_BASE_URL}.`, 0);
  }

  if (!response.ok) {
    const detail = await parseErrorMessage(response);
    throw new ApiError(detail, response.status);
  }

  return response.json();
}

/**
 * Upload an image frame Blob to POST /detect and return structured detections.
 * @param {Blob} imageBlob - Encoded JPEG/PNG/WebP image frame
 * @param {number} confidence - Confidence threshold between 0.01 and 1.0
 * @param {AbortSignal} [signal]
 */
export async function detectObjects(imageBlob, confidence = 0.35, signal) {
  const formData = new FormData();
  formData.append("file", imageBlob, "frame.jpg");

  const clampedConfidence = Math.min(1.0, Math.max(0.01, Number(confidence) || 0.35));
  const url = `${API_BASE_URL}/detect?confidence=${encodeURIComponent(
    clampedConfidence.toFixed(2)
  )}`;

  let response;
  try {
    response = await fetch(url, {
      method: "POST",
      body: formData,
      signal,
    });
  } catch (err) {
    if (err.name === "AbortError") throw err;
    throw new ApiError(
      "Network error while sending frame to backend. Retrying on next frame...",
      0
    );
  }

  if (!response.ok) {
    const detail = await parseErrorMessage(response);
    throw new ApiError(detail, response.status);
  }

  return response.json();
}
