import React from "react";

export function ControlPanel({
  cameraStatus,
  isDetecting,
  confidence,
  onConfidenceChange,
  captureIntervalMs,
  onIntervalChange,
  detectionCount,
  inferenceTimeMs,
  roundTripTimeMs,
  videoDimensions,
  onStartCamera,
  onStopCamera,
  onToggleDetection,
}) {
  const isCameraActive = cameraStatus === "active";
  const isCameraStarting = cameraStatus === "starting";

  return (
    <section className="control-panel-card" aria-label="Detection controls and metrics">
      <div className="controls-row">
        <div className="button-group">
          {!isCameraActive ? (
            <button
              type="button"
              className="btn btn-primary"
              onClick={onStartCamera}
              disabled={isCameraStarting}
            >
              {isCameraStarting ? "Starting Camera..." : "Start Camera"}
            </button>
          ) : (
            <button
              type="button"
              className="btn btn-danger"
              onClick={onStopCamera}
            >
              Stop Camera
            </button>
          )}

          <button
            type="button"
            className={`btn ${isDetecting ? "btn-warning" : "btn-secondary"}`}
            onClick={onToggleDetection}
            disabled={!isCameraActive}
          >
            {isDetecting ? "Pause Detection" : "Resume Detection"}
          </button>
        </div>

        <div className="sliders-group">
          <div className="slider-control">
            <div className="slider-label-row">
              <label htmlFor="confidence-slider">Confidence Threshold</label>
              <span className="slider-value">
                {Math.round(confidence * 100)}% ({confidence.toFixed(2)})
              </span>
            </div>
            <input
              id="confidence-slider"
              type="range"
              min="0.10"
              max="0.95"
              step="0.05"
              value={confidence}
              onChange={(e) => onConfidenceChange( parseFloat(e.target.value) )}
            />
          </div>

          <div className="interval-control">
            <label htmlFor="interval-select">Frame Cadence</label>
            <select
              id="interval-select"
              value={captureIntervalMs}
              onChange={(e) => onIntervalChange(Number(e.target.value))}
            >
              <option value={200}>200 ms (~5 FPS)</option>
              <option value={250}>250 ms (4 FPS)</option>
              <option value={300}>300 ms (~3.3 FPS)</option>
              <option value={500}>500 ms (2 FPS)</option>
            </select>
          </div>
        </div>
      </div>

      <div className="telemetry-grid">
        <div className="telemetry-card">
          <span className="telemetry-label">Detected Objects</span>
          <strong className="telemetry-value telemetry-accent">
            {detectionCount}
          </strong>
        </div>

        <div className="telemetry-card">
          <span className="telemetry-label">YOLO Inference Latency</span>
          <strong className="telemetry-value">
            {inferenceTimeMs !== null ? `${inferenceTimeMs.toFixed(1)} ms` : "—"}
          </strong>
        </div>

        <div className="telemetry-card">
          <span className="telemetry-label">Total Frame RTT</span>
          <strong className="telemetry-value">
            {roundTripTimeMs !== null ? `${roundTripTimeMs} ms` : "—"}
          </strong>
        </div>

        <div className="telemetry-card">
          <span className="telemetry-label">Camera Resolution</span>
          <strong className="telemetry-value">
            {videoDimensions.width > 0
              ? `${videoDimensions.width} × ${videoDimensions.height}`
              : "—"}
          </strong>
        </div>
      </div>
    </section>
  );
}
