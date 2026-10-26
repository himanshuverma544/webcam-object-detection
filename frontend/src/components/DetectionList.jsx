import React, { useState } from "react";
import { getClassColor } from "./VideoViewport";

export function DetectionList({
  detections,
  detectionCount,
  inferenceTimeMs,
  sourceFrameSize,
  cameraStatus,
  isDetecting,
  supportedClasses,
  lastUpdatedAt,
}) {
  const [showClassesDrawer, setShowClassesDrawer] = useState(false);

  return (
    <aside className="detection-sidebar-card" aria-label="Detected objects list">
      <div className="sidebar-header">
        <div>
          <h2>Detected Objects</h2>
          <p className="sidebar-subtitle">
            {inferenceTimeMs !== null
              ? `Inference: ${inferenceTimeMs.toFixed(1)} ms • Frame: ${sourceFrameSize.width}×${sourceFrameSize.height}px`
              : "Live frame detections & bounding boxes"}
          </p>
        </div>
        <span className="count-badge" title="Total detected objects in current frame">
          {detectionCount}
        </span>
      </div>

      {lastUpdatedAt && (
        <div className="sidebar-timestamp">
          Last updated: {lastUpdatedAt.toLocaleTimeString()}
        </div>
      )}

      <div className="detections-scroll-area">
        {cameraStatus !== "active" ? (
          <div className="empty-detections-state">
            <p>Start the camera to detect objects in real time.</p>
            <span className="empty-hint">
              Detects 80 COCO object categories including people, bottles, cups,
              cell phones, laptops, chairs, books, and cars.
            </span>
          </div>
        ) : detections.length === 0 ? (
          <div className="empty-detections-state">
            <p>
              {isDetecting
                ? "No objects detected above current confidence threshold."
                : "Detection is currently paused."}
            </p>
            <span className="empty-hint">
              Try holding up a phone, cup, bottle, or book, or lower the
              confidence slider.
            </span>
          </div>
        ) : (
          <ul className="detection-items-list">
            {detections.map((item, idx) => {
              const color = getClassColor(item.class_id);
              const pct = ((item.confidence || 0) * 100).toFixed(1);
              const [x1, y1, x2, y2] = (item.bbox || [0, 0, 0, 0]).map((v) =>
                Math.round(v)
              );

              return (
                <li
                  key={`${item.class_id}-${idx}-${x1}-${y1}`}
                  className="detection-item-card"
                >
                  <div className="detection-item-top">
                    <div className="detection-label-group">
                      <span
                        className="detection-color-swatch"
                        style={{ backgroundColor: color }}
                      />
                      <span className="detection-label-name">{item.label}</span>
                      <span className="detection-class-id">
                        ID #{item.class_id}
                      </span>
                    </div>
                    <strong className="detection-confidence-score">
                      {pct}%
                    </strong>
                  </div>

                  <div className="confidence-bar-track">
                    <div
                      className="confidence-bar-fill"
                      style={{
                        width: `${Math.min(100, Math.max(2, Number(pct)))}%`,
                        backgroundColor: color,
                      }}
                    />
                  </div>

                  <div className="detection-bbox-coords">
                    <span className="bbox-tag">BBox [x1, y1, x2, y2]:</span>
                    <code>
                      [{x1}, {y1}, {x2}, {y2}]
                    </code>
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </div>

      <div className="classes-footer">
        <button
          type="button"
          className="classes-toggle-btn"
          onClick={() => setShowClassesDrawer((prev) => !prev)}
        >
          <span>
            Supported COCO Classes (
            {supportedClasses.length > 0 ? supportedClasses.length : 80})
          </span>
          <span>{showClassesDrawer ? "▲" : "▼"}</span>
        </button>

        {showClassesDrawer && (
          <div className="classes-pill-container">
            {(supportedClasses.length > 0
              ? supportedClasses
              : [
                  "person",
                  "bicycle",
                  "car",
                  "bottle",
                  "cup",
                  "chair",
                  "couch",
                  "laptop",
                  "mouse",
                  "keyboard",
                  "cell phone",
                  "book",
                  "clock",
                  "backpack",
                ]
            ).map((clsName) => (
              <span key={clsName} className="coco-class-pill">
                {clsName}
              </span>
            ))}
          </div>
        )}
      </div>
    </aside>
  );
}
