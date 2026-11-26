const rawBaseUrl =
  import.meta.env.VITE_API_BASE_URL !== undefined
    ? import.meta.env.VITE_API_BASE_URL
    : import.meta.env.DEV
      ? "http://127.0.0.1:8000"
      : "";

// Ensure no trailing slashes and auto-upgrade http to https if loaded over HTTPS (prevents Mixed Content errors)
let normalizedUrl = (rawBaseUrl || "").trim().replace(/\/+$/, "");
if (
  typeof window !== "undefined" &&
  window.location.protocol === "https:" &&
  normalizedUrl.startsWith("http://") &&
  !normalizedUrl.includes("localhost") &&
  !normalizedUrl.includes("127.0.0.1")
) {
  normalizedUrl = normalizedUrl.replace(/^http:\/\//, "https://");
}

export const API_BASE_URL = normalizedUrl;

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
 * @param {string[]|null|AbortSignal} [classes] - Optional array of selected COCO class names or IDs
 * @param {AbortSignal} [signal]
 */
export async function detectObjects(
  imageBlob,
  confidence = 0.35,
  classes = null,
  signal,
  customItemIds = null,
  similarityThreshold = null
) {
  let activeClasses = classes;
  let activeSignal = signal;
  if (classes && (classes instanceof AbortSignal || typeof classes.aborted === "boolean")) {
    activeSignal = classes;
    activeClasses = null;
  }

  const formData = new FormData();
  formData.append("file", imageBlob, "frame.jpg");

  const clampedConfidence = Math.min(1.0, Math.max(0.01, Number(confidence) || 0.35));
  let url = `${API_BASE_URL}/detect?confidence=${encodeURIComponent(
    clampedConfidence.toFixed(2)
  )}`;

  const hasCustom = Array.isArray(customItemIds) && customItemIds.length > 0;
  const hasCoco = Array.isArray(activeClasses) && activeClasses.length > 0;

  if (hasCoco) {
    url += `&classes=${encodeURIComponent(activeClasses.join(","))}`;
  } else if (hasCustom) {
    // Explicitly signal 0 COCO classes so backend skips YOLO11n and runs custom YOLOE only
    url += `&classes=`;
  }

  if (hasCustom) {
    url += `&custom_items=${encodeURIComponent(customItemIds.join(","))}`;
    if (typeof similarityThreshold === "number" && !Number.isNaN(similarityThreshold)) {
      const clampedSim = Math.min(1.0, Math.max(0.1, similarityThreshold));
      url += `&similarity_threshold=${encodeURIComponent(clampedSim.toFixed(2))}`;
    }
  }

  let response;
  try {
    response = await fetch(url, {
      method: "POST",
      body: formData,
      signal: activeSignal,
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

/**
 * Reset tracker states and object tracking IDs in the backend.
 * @param {AbortSignal} [signal]
 */
export async function resetTrack(signal) {
  try {
    const response = await fetch(`${API_BASE_URL}/track/reset`, {
      method: "POST",
      headers: { Accept: "application/json" },
      signal,
    });
    if (response.ok) {
      return await response.json();
    }
  } catch (err) {
    if (err.name === "AbortError") throw err;
    // Non-fatal if tracker reset fails
  }
  return null;
}

const defaultEnvRecipient = (import.meta.env.VITE_DEFAULT_RECIPIENT_EMAIL || "").trim();

/**
 * Dispatch session metrics report via email to specified recipient.
 * @param {Object} metrics - Structured session metrics object
 * @param {string} [recipientEmail] - Destination email (falls back to VITE_DEFAULT_RECIPIENT_EMAIL if omitted)
 * @param {AbortSignal} [signal]
 */
export async function sendMetricsEmail(
  metrics,
  recipientEmail = defaultEnvRecipient,
  signal
) {
  let response;
  try {
    const payload = { metrics };
    if (recipientEmail && recipientEmail.trim()) {
      payload.recipient_email = recipientEmail.trim();
    }

    response = await fetch(`${API_BASE_URL}/metrics/email`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify(payload),
      signal,
    });
  } catch (err) {
    if (err.name === "AbortError") throw err;
    throw new ApiError(
      `Cannot reach backend at ${API_BASE_URL}. Ensure the server is running.`,
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
 * Dispatch session metrics summary via SMS to specified recipient.
 * @param {Object} metrics - Structured session metrics object
 * @param {string} [recipientPhone] - Destination phone (falls back to backend RECIPIENT_PHONE_NUMBER if omitted)
 * @param {AbortSignal} [signal]
 */
export async function sendMetricsSMS(
  metrics,
  recipientPhone,
  signal
) {
  let response;
  try {
    const payload = { metrics };
    if (recipientPhone && recipientPhone.trim()) {
      payload.recipient_phone = recipientPhone.trim();
    }

    response = await fetch(`${API_BASE_URL}/metrics/sms`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify(payload),
      signal,
    });
  } catch (err) {
    if (err.name === "AbortError") throw err;
    throw new ApiError(
      `Cannot reach backend at ${API_BASE_URL}. Ensure the server is running.`,
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
 * Fetch all saved custom visual-prompt reference items.
 * @param {AbortSignal} [signal]
 */
export async function fetchCustomItems(signal) {
  let response;
  try {
    response = await fetch(`${API_BASE_URL}/custom-items`, {
      method: "GET",
      headers: { Accept: "application/json" },
      signal,
    });
  } catch (err) {
    if (err.name === "AbortError") throw err;
    throw new ApiError(
      `Cannot reach backend at ${API_BASE_URL}. Ensure the server is running.`,
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
 * Auto-detect a tight foreground bounding box around the main object in a photo.
 * @param {File|Blob} file
 */
export async function autoBoxCustomImage(file) {
  const formData = new FormData();
  const filename = file instanceof File && file.name ? file.name : "ref.jpg";
  formData.append("file", file, filename);

  let response;
  try {
    response = await fetch(`${API_BASE_URL}/custom-items/auto-box`, {
      method: "POST",
      body: formData,
    });
  } catch (err) {
    throw new ApiError(
      `Cannot reach backend at ${API_BASE_URL}. Ensure the server is running.`,
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
 * Register a new custom reference item from 1-5 image files with optional tight foreground boxes.
 * @param {string} name
 * @param {Array<File|Blob>} files
 * @param {Array<Array<number>>} [bboxes] - Normalized [x1, y1, x2, y2] boxes matching files
 */
export async function createCustomItem(name, files, bboxes = null) {
  const formData = new FormData();
  formData.append("name", name.trim());
  files.forEach((f, idx) => {
    const filename = f instanceof File && f.name ? f.name : `ref_${idx + 1}.jpg`;
    formData.append("files", f, filename);
  });
  if (Array.isArray(bboxes) && bboxes.length > 0) {
    formData.append("bboxes", JSON.stringify(bboxes));
  }

  let response;
  try {
    response = await fetch(`${API_BASE_URL}/custom-items`, {
      method: "POST",
      body: formData,
    });
  } catch (err) {
    throw new ApiError(
      `Cannot reach backend at ${API_BASE_URL}. Ensure the server is running.`,
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
 * Add extra reference angle photos (with optional tight boxes) to an existing custom item.
 * @param {string} itemId
 * @param {Array<File|Blob>} files
 * @param {Array<Array<number>>} [bboxes]
 */
export async function addCustomItemImages(itemId, files, bboxes = null) {
  const formData = new FormData();
  files.forEach((f, idx) => {
    const filename = f instanceof File && f.name ? f.name : `angle_${idx + 1}.jpg`;
    formData.append("files", f, filename);
  });
  if (Array.isArray(bboxes) && bboxes.length > 0) {
    formData.append("bboxes", JSON.stringify(bboxes));
  }

  let response;
  try {
    response = await fetch(
      `${API_BASE_URL}/custom-items/${encodeURIComponent(itemId)}/images`,
      {
        method: "POST",
        body: formData,
      }
    );
  } catch (err) {
    throw new ApiError(
      `Cannot reach backend at ${API_BASE_URL}. Ensure the server is running.`,
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
 * Fetch complete detail for a single custom item including all reference photos.
 * @param {string} itemId
 */
export async function fetchCustomItemDetail(itemId) {
  let response;
  try {
    response = await fetch(
      `${API_BASE_URL}/custom-items/${encodeURIComponent(itemId)}`,
      {
        method: "GET",
        headers: { Accept: "application/json" },
      }
    );
  } catch (err) {
    throw new ApiError(
      `Cannot reach backend at ${API_BASE_URL}. Ensure the server is running.`,
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
 * Update an existing custom item (name, retained photos, new files).
 * @param {string} itemId
 * @param {Object} options
 * @param {string} [options.name]
 * @param {Array<{file_name: string, bbox_norm: number[]}>} [options.retainedPhotos]
 * @param {File[]} [options.newFiles]
 * @param {Array<number[]>} [options.newBboxes]
 */
export async function updateCustomItem(itemId, { name, retainedPhotos = null, newFiles = [], newBboxes = null }) {
  const formData = new FormData();
  if (name) {
    formData.append("name", name);
  }
  if (Array.isArray(retainedPhotos)) {
    formData.append("retained_photos", JSON.stringify(retainedPhotos));
  }
  if (Array.isArray(newFiles)) {
    for (const f of newFiles) {
      formData.append("files", f);
    }
  }
  if (Array.isArray(newBboxes) && newBboxes.length > 0) {
    formData.append("bboxes", JSON.stringify(newBboxes));
  }

  let response;
  try {
    response = await fetch(
      `${API_BASE_URL}/custom-items/${encodeURIComponent(itemId)}`,
      {
        method: "PUT",
        body: formData,
      }
    );
  } catch (err) {
    throw new ApiError(
      `Cannot reach backend at ${API_BASE_URL}. Ensure the server is running.`,
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
 * Delete a saved custom reference item by ID.
 * @param {string} itemId
 */
export async function deleteCustomItem(itemId) {
  let response;
  try {
    response = await fetch(
      `${API_BASE_URL}/custom-items/${encodeURIComponent(itemId)}`,
      {
        method: "DELETE",
        headers: { Accept: "application/json" },
      }
    );
  } catch (err) {
    throw new ApiError(
      `Cannot reach backend at ${API_BASE_URL}. Ensure the server is running.`,
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
 * Send a chat instruction or multimodal message to the AI Copilot.
 * @param {Object} options
 * @param {string} options.message - Natural language command
 * @param {Array} [options.history] - Prior messages [{ role, content }]
 * @param {Object} [options.currentState] - Current UI state
 * @param {File|Blob|null} [options.file] - Optional uploaded photo
 * @param {AbortSignal} [options.signal]
 */
export async function sendChatMessage({
  message,
  history = [],
  currentState = {},
  file = null,
  signal,
}) {
  let response;
  try {
    if (file) {
      const formData = new FormData();
      formData.append("message", message || "");
      formData.append("file", file, file.name || "upload.jpg");
      if (history && history.length > 0) {
        formData.append("history", JSON.stringify(history));
      }
      if (currentState && Object.keys(currentState).length > 0) {
        formData.append("current_state", JSON.stringify(currentState));
      }

      response = await fetch(`${API_BASE_URL}/chat`, {
        method: "POST",
        body: formData,
        signal,
      });
    } else {
      response = await fetch(`${API_BASE_URL}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify({
          message,
          history,
          current_state: currentState,
        }),
        signal,
      });
    }
  } catch (err) {
    if (err.name === "AbortError") throw err;
    throw new ApiError(
      `Cannot reach AI Copilot backend at ${API_BASE_URL}. Ensure the server is running.`,
      0
    );
  }

  if (!response.ok) {
    const detail = await parseErrorMessage(response);
    throw new ApiError(detail, response.status);
  }

  return response.json();
}
