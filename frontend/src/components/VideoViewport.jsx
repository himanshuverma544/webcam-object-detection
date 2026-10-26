import React, { useCallback, useEffect, useRef } from "react";

// High-contrast color palette indexed by COCO class_id
const BOX_PALETTE = [
  "#38bdf8", // sky blue
  "#34d399", // emerald
  "#f472b6", // pink
  "#fbbf24", // amber
  "#a78bfa", // violet
  "#fb7185", // rose
  "#4ade80", // green
  "#60a5fa", // blue
  "#f97316", // orange
  "#2dd4bf", // teal
];

export function getClassColor(classId = 0) {
  const index = Math.abs(Number(classId) || 0) % BOX_PALETTE.length;
  return BOX_PALETTE[index];
}

export function VideoViewport({
  videoRef,
  cameraStatus,
  cameraError,
  isDetecting,
  detections,
  sourceFrameSize,
  onStartCamera,
  onMetadataLoaded,
}) {
  const containerRef = useRef(null);
  const overlayCanvasRef = useRef(null);

  const drawOverlay = useCallback(() => {
    const canvas = overlayCanvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const rect = video.getBoundingClientRect();
    const displayWidth = Math.round(rect.width);
    const displayHeight = Math.round(rect.height);

    if (displayWidth <= 0 || displayHeight <= 0) {
      return;
    }

    const dpr = window.devicePixelRatio || 1;
    if (
      canvas.width !== Math.round(displayWidth * dpr) ||
      canvas.height !== Math.round(displayHeight * dpr)
    ) {
      canvas.width = Math.round(displayWidth * dpr);
      canvas.height = Math.round(displayHeight * dpr);
      canvas.style.width = `${displayWidth}px`;
      canvas.style.height = `${displayHeight}px`;
    }

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, displayWidth, displayHeight);

    if (
      cameraStatus !== "active" ||
      !Array.isArray(detections) ||
      detections.length === 0
    ) {
      ctx.restore();
      return;
    }

    const srcWidth =
      sourceFrameSize?.width || video.videoWidth || displayWidth;
    const srcHeight =
      sourceFrameSize?.height || video.videoHeight || displayHeight;

    if (srcWidth <= 0 || srcHeight <= 0) {
      ctx.restore();
      return;
    }

    // Compute actual rendered video content box inside the <video> element (object-fit: contain)
    const videoAspect = srcWidth / srcHeight;
    const elementAspect = displayWidth / displayHeight;

    let renderWidth = displayWidth;
    let renderHeight = displayHeight;
    let offsetX = 0;
    let offsetY = 0;

    if (elementAspect > videoAspect) {
      // Pillarbox (horizontal bars)
      renderHeight = displayHeight;
      renderWidth = displayHeight * videoAspect;
      offsetX = (displayWidth - renderWidth) / 2;
    } else {
      // Letterbox (vertical bars)
      renderWidth = displayWidth;
      renderHeight = displayWidth / videoAspect;
      offsetY = (displayHeight - renderHeight) / 2;
    }

    const scaleX = renderWidth / srcWidth;
    const scaleY = renderHeight / srcHeight;

    ctx.font = "600 13px Inter, -apple-system, BlinkMacSystemFont, sans-serif";
    ctx.textBaseline = "top";

    detections.forEach((det) => {
      if (!Array.isArray(det.bbox) || det.bbox.length < 4) return;
      const [rawX1, rawY1, rawX2, rawY2] = det.bbox;

      const x = offsetX + rawX1 * scaleX;
      const y = offsetY + rawY1 * scaleY;
      const w = Math.max(4, (rawX2 - rawX1) * scaleX);
      const h = Math.max(4, (rawY2 - rawY1) * scaleY);

      const color = getClassColor(det.class_id);
      const confidencePct = `${Math.round((det.confidence || 0) * 100)}%`;
      const labelText = `${det.label} ${confidencePct}`;

      // Semi-transparent fill
      ctx.fillStyle = `${color}1A`;
      ctx.fillRect(x, y, w, h);

      // Crisp bounding box outline
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.strokeRect(x, y, w, h);

      // Measure and draw label pill
      const textMetrics = ctx.measureText(labelText);
      const pillPaddingX = 8;
      const pillHeight = 22;
      const pillWidth = Math.min(
        renderWidth,
        textMetrics.width + pillPaddingX * 2
      );

      // Place pill above box unless too close to top edge
      const pillX = Math.max(
        offsetX,
        Math.min(x, offsetX + renderWidth - pillWidth)
      );
      const pillY = y - pillHeight >= offsetY ? y - pillHeight : y;

      ctx.fillStyle = color;
      ctx.beginPath();
      if (typeof ctx.roundRect === "function") {
        ctx.roundRect(pillX, pillY, pillWidth, pillHeight, 4);
      } else {
        ctx.rect(pillX, pillY, pillWidth, pillHeight);
      }
      ctx.fill();

      // Draw high-contrast label text
      ctx.fillStyle = "#090d16";
      ctx.fillText(labelText, pillX + pillPaddingX, pillY + 4);
    });

    ctx.restore();
  }, [cameraStatus, detections, sourceFrameSize, videoRef]);

  // Redraw whenever detections or frame dimensions update
  useEffect(() => {
    drawOverlay();
  }, [drawOverlay]);

  // Keep bounding boxes aligned when the video container or window resizes
  useEffect(() => {
    const videoEl = videoRef.current;
    const containerEl = containerRef.current;

    const handleResize = () => {
      drawOverlay();
    };

    window.addEventListener("resize", handleResize);

    let resizeObserver;
    if (typeof ResizeObserver !== "undefined") {
      resizeObserver = new ResizeObserver(() => {
        drawOverlay();
      });
      if (videoEl) resizeObserver.observe(videoEl);
      if (containerEl) resizeObserver.observe(containerEl);
    }

    return () => {
      window.removeEventListener("resize", handleResize);
      if (resizeObserver) {
        resizeObserver.disconnect();
      }
    };
  }, [drawOverlay, videoRef]);

  return (
    <div className="video-viewport-card" ref={containerRef}>
      <div className="video-stage">
        <video
          ref={videoRef}
          className={`webcam-video ${
            cameraStatus === "active" ? "video-visible" : "video-hidden"
          }`}
          autoPlay
          playsInline
          muted
          onLoadedMetadata={() => {
            onMetadataLoaded();
            drawOverlay();
          }}
        />

        <canvas
          ref={overlayCanvasRef}
          className="bbox-overlay-canvas"
          aria-label="Object detection bounding box overlay"
        />

        {cameraStatus === "active" && !isDetecting && (
          <div className="viewport-badge viewport-badge-paused">
            Detection Paused — Live Camera Active
          </div>
        )}

        {cameraStatus === "idle" && (
          <div className="viewport-placeholder">
            <div className="placeholder-icon" aria-hidden="true">
              <svg
                width="48"
                height="48"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
                <circle cx="12" cy="13" r="4" />
              </svg>
            </div>
            <h2>Webcam Feed Inactive</h2>
            <p>
              Click <strong>Start Camera</strong> to grant webcam permission and
              begin real-time YOLO11n object detection.
            </p>
            <button
              type="button"
              className="btn btn-primary btn-lg"
              onClick={onStartCamera}
            >
              Start Camera
            </button>
          </div>
        )}

        {cameraStatus === "starting" && (
          <div className="viewport-placeholder">
            <div className="spinner" aria-hidden="true" />
            <h2>Requesting Camera Access...</h2>
            <p>Please allow webcam access in your browser prompt.</p>
          </div>
        )}

        {cameraStatus === "error" && (
          <div className="viewport-placeholder viewport-error" role="alert">
            <div className="error-icon" aria-hidden="true">
              !
            </div>
            <h2>Camera Unavailable</h2>
            <p>{cameraError}</p>
            <button
              type="button"
              className="btn btn-primary"
              onClick={onStartCamera}
            >
              Try Again
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
