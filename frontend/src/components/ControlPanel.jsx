import React from "react";

export function ControlPanel({
  cameraStatus,
  isDetecting,
  confidence,
  onConfidenceChange,
  captureIntervalMs,
  onIntervalChange,
  minMoveDistance,
  onMinMoveDistanceChange,
  settleDwellTimeMs,
  onSettleDwellTimeMsChange,
  activeMovingCount = 0,
  movementEventsCount = 0,
  isGuarded = false,
  guardedCount = 0,
  onToggleGuard,
  missingCount = 0,
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

          <button
            type="button"
            className={`btn ${isGuarded ? "btn-armed" : "btn-guard"}`}
            onClick={onToggleGuard}
            disabled={!isCameraActive}
            title={isGuarded ? "Disarm guarded scene baseline" : "Lock currently detected items as guarded baseline"}
          >
            {isGuarded ? `🔓 Disarm Scene (${guardedCount})` : `🛡️ Guard Scene (${detectionCount})`}
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

          <div className="slider-control">
            <div className="slider-label-row">
              <label htmlFor="min-dist-slider">Min Move Distance</label>
              <span className="slider-value">{minMoveDistance} px</span>
            </div>
            <input
              id="min-dist-slider"
              type="range"
              min="25"
              max="150"
              step="5"
              value={minMoveDistance}
              onChange={(e) => onMinMoveDistanceChange(Number(e.target.value))}
            />
          </div>

          <div className="slider-control">
            <div className="slider-label-row">
              <label htmlFor="dwell-slider">Settle Dwell Time</label>
              <span className="slider-value">{(settleDwellTimeMs / 1000).toFixed(1)}s</span>
            </div>
            <input
              id="dwell-slider"
              type="range"
              min="800"
              max="3000"
              step="200"
              value={settleDwellTimeMs}
              onChange={(e) => onSettleDwellTimeMsChange(Number(e.target.value))}
            />
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
          <span className="telemetry-label">Items Moving</span>
          <strong className={`telemetry-value ${activeMovingCount > 0 ? "telemetry-moving" : ""}`}>
            {activeMovingCount > 0 ? `${activeMovingCount} active` : "0"}
          </strong>
        </div>

        <div className="telemetry-card">
          <span className="telemetry-label">Total Relocations</span>
          <strong className="telemetry-value telemetry-success">
            {movementEventsCount}
          </strong>
        </div>

        <div className="telemetry-card">
          <span className="telemetry-label">Missing Items</span>
          <strong className={`telemetry-value ${missingCount > 0 ? "telemetry-missing" : ""}`}>
            {missingCount > 0 ? `🚨 ${missingCount}` : "0"}
          </strong>
        </div>

        <div className="telemetry-card">
          <span className="telemetry-label">YOLO Inference</span>
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
