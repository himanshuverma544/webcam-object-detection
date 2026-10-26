import React from "react";
import { API_BASE_URL } from "../services/api";

export function Header({
  backendStatus,
  healthInfo,
  detectionStatus,
  onRetryHealth,
}) {
  const getBackendBadge = () => {
    switch (backendStatus) {
      case "ready":
        return {
          label: `Backend Ready (${healthInfo?.model_name || "yolo11n.pt"} • ${(
            healthInfo?.device || "cpu"
          ).toUpperCase()})`,
          className: "status-pill status-ready",
        };
      case "loading":
        return {
          label: "Model Initializing...",
          className: "status-pill status-warning",
        };
      case "connecting":
        return {
          label: "Connecting to Backend...",
          className: "status-pill status-neutral",
        };
      default:
        return {
          label: "Backend Disconnected",
          className: "status-pill status-error",
        };
    }
  };

  const getDetectionBadge = () => {
    switch (detectionStatus) {
      case "running":
        return {
          label: "Detection Active",
          className: "status-pill status-active",
        };
      case "paused":
        return {
          label: "Detection Paused",
          className: "status-pill status-warning",
        };
      case "error":
        return {
          label: "Inference Warning",
          className: "status-pill status-error",
        };
      default:
        return {
          label: "Detection Idle",
          className: "status-pill status-neutral",
        };
    }
  };

  const backendBadge = getBackendBadge();
  const detectionBadge = getDetectionBadge();

  return (
    <header className="dashboard-header">
      <div className="header-brand">
        <div className="brand-title-row">
          <span className="brand-icon" aria-hidden="true">
            <svg
              width="24"
              height="24"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z" />
              <circle cx="12" cy="13" r="3" />
            </svg>
          </span>
          <h1>YOLO11 Real-Time Object Detection</h1>
        </div>
        <p className="brand-subtitle">
          Live webcam computer vision powered by Ultralytics YOLO11n, OpenCV &amp;
          FastAPI ({API_BASE_URL})
        </p>
      </div>

      <div className="header-statuses">
        <div
          className={backendBadge.className}
          title={healthInfo?.error || `Connected to ${API_BASE_URL}`}
        >
          <span className="status-dot" />
          <span>{backendBadge.label}</span>
          {backendStatus === "disconnected" && (
            <button
              type="button"
              className="status-retry-btn"
              onClick={onRetryHealth}
            >
              Retry
            </button>
          )}
        </div>

        <div className={detectionBadge.className}>
          <span className="status-dot" />
          <span>{detectionBadge.label}</span>
        </div>
      </div>
    </header>
  );
}
