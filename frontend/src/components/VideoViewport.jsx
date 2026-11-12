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
  activeTracksMap = {},
  missingItems = [],
  isGuarded = false,
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

    // 1. Draw Ghost Origin Boxes and Trajectory Lines for moving or recently moved items
    if (activeTracksMap && typeof activeTracksMap === "object") {
      Object.values(activeTracksMap).forEach((track) => {
        if (!track || (!track.isFlashing && track.state !== "MOVING")) return;

        const color = getClassColor(track.class_id);
        const originBbox = track.origin?.bbox;

        // Draw Ghost Origin Bounding Box
        if (Array.isArray(originBbox) && originBbox.length >= 4) {
          const [ox1, oy1, ox2, oy2] = originBbox;
          const gx = offsetX + ox1 * scaleX;
          const gy = offsetY + oy1 * scaleY;
          const gw = Math.max(4, (ox2 - ox1) * scaleX);
          const gh = Math.max(4, (oy2 - oy1) * scaleY);

          ctx.save();
          ctx.setLineDash([5, 4]);
          ctx.strokeStyle = track.isFlashing ? "#10b98188" : `${color}99`;
          ctx.lineWidth = 2;
          ctx.strokeRect(gx, gy, gw, gh);
          ctx.fillStyle = `${color}0D`;
          ctx.fillRect(gx, gy, gw, gh);

          // Small origin indicator label
          const originLabel = "Origin";
          ctx.font = "600 10px Inter, -apple-system, BlinkMacSystemFont, sans-serif";
          const om = ctx.measureText(originLabel);
          const opW = om.width + 8;
          const opH = 16;
          ctx.fillStyle = track.isFlashing ? "#10b981" : `${color}CC`;
          if (typeof ctx.roundRect === "function") {
            ctx.roundRect(gx, Math.max(offsetY, gy - opH), opW, opH, 3);
          } else {
            ctx.rect(gx, Math.max(offsetY, gy - opH), opW, opH);
          }
          ctx.fill();
          ctx.fillStyle = "#ffffff";
          ctx.fillText(originLabel, gx + 4, Math.max(offsetY, gy - opH) + 2);
          ctx.restore();
        }

        // Draw Trajectory Path and Arrowhead
        const path = track.path;
        if (Array.isArray(path) && path.length >= 2) {
          ctx.save();
          ctx.beginPath();
          const startX = offsetX + path[0].x * scaleX;
          const startY = offsetY + path[0].y * scaleY;
          ctx.moveTo(startX, startY);

          for (let i = 1; i < path.length; i++) {
            const px = offsetX + path[i].x * scaleX;
            const py = offsetY + path[i].y * scaleY;
            ctx.lineTo(px, py);
          }

          ctx.strokeStyle = track.isFlashing ? "#10b981" : color;
          ctx.lineWidth = track.isFlashing ? 3.5 : 2.5;
          ctx.lineCap = "round";
          ctx.lineJoin = "round";
          ctx.stroke();

          // Draw breadcrumb points along the trajectory
          path.forEach((pt, idx) => {
            const px = offsetX + pt.x * scaleX;
            const py = offsetY + pt.y * scaleY;
            const alpha = 0.3 + 0.7 * (idx / path.length);
            ctx.fillStyle = track.isFlashing ? `rgba(16, 185, 129, ${alpha})` : color;
            ctx.beginPath();
            ctx.arc(px, py, idx === path.length - 1 ? 4 : 2.5, 0, Math.PI * 2);
            ctx.fill();
          });

          // Draw Arrowhead at destination
          const last = path[path.length - 1];
          const prev = path[Math.max(0, path.length - 2)];
          const angle = Math.atan2((last.y - prev.y) * scaleY, (last.x - prev.x) * scaleX);
          const arrowLength = 12;
          const endX = offsetX + last.x * scaleX;
          const endY = offsetY + last.y * scaleY;

          ctx.fillStyle = track.isFlashing ? "#10b981" : color;
          ctx.beginPath();
          ctx.moveTo(endX, endY);
          ctx.lineTo(
            endX - arrowLength * Math.cos(angle - Math.PI / 6),
            endY - arrowLength * Math.sin(angle - Math.PI / 6)
          );
          ctx.lineTo(
            endX - arrowLength * Math.cos(angle + Math.PI / 6),
            endY - arrowLength * Math.sin(angle + Math.PI / 6)
          );
          ctx.closePath();
          ctx.fill();

          // Distance badge at midpoint
          if (track.distance > 0) {
            const midX = (startX + endX) / 2;
            const midY = (startY + endY) / 2;
            const distText = track.isFlashing ? `Moved ${track.distance}px ✓` : `${track.distance}px ➔`;
            ctx.font = "600 11px Inter, -apple-system, BlinkMacSystemFont, sans-serif";
            const dm = ctx.measureText(distText);
            const badgeW = dm.width + 10;
            const badgeH = 18;

            ctx.fillStyle = "rgba(9, 13, 22, 0.85)";
            if (typeof ctx.roundRect === "function") {
              ctx.roundRect(midX - badgeW / 2, midY - badgeH / 2, badgeW, badgeH, 4);
            } else {
              ctx.rect(midX - badgeW / 2, midY - badgeH / 2, badgeW, badgeH);
            }
            ctx.fill();
            ctx.strokeStyle = track.isFlashing ? "#10b981" : color;
            ctx.lineWidth = 1;
            ctx.stroke();

            ctx.fillStyle = track.isFlashing ? "#34d399" : "#ffffff";
            ctx.fillText(distText, midX - badgeW / 2 + 5, midY - badgeH / 2 + 2);
          }

          ctx.restore();
        }
      });
    }

    ctx.font = "600 13px Inter, -apple-system, BlinkMacSystemFont, sans-serif";
    ctx.textBaseline = "top";

    detections.forEach((det) => {
      if (!Array.isArray(det.bbox) || det.bbox.length < 4) return;
      const [rawX1, rawY1, rawX2, rawY2] = det.bbox;

      const x = offsetX + rawX1 * scaleX;
      const y = offsetY + rawY1 * scaleY;
      const w = Math.max(4, (rawX2 - rawX1) * scaleX);
      const h = Math.max(4, (rawY2 - rawY1) * scaleY);

      // Match track in activeTracksMap
      let trackInfo = null;
      if (activeTracksMap && typeof activeTracksMap === "object") {
        const cx = Math.round((rawX1 + rawX2) / 2);
        const cy = Math.round((rawY1 + rawY2) / 2);

        if (det.track_id !== undefined && det.track_id !== null) {
          trackInfo = Object.values(activeTracksMap).find(
            (t) => t.track_id === det.track_id && t.class_id === det.class_id
          );
        }

        if (!trackInfo) {
          let closestDist = 90;
          for (const t of Object.values(activeTracksMap)) {
            if (t.class_id === det.class_id) {
              const d = Math.hypot(cx - t.current.x, cy - t.current.y);
              if (d < closestDist) {
                closestDist = d;
                trackInfo = t;
              }
            }
          }
        }
      }

      const isMoving = trackInfo?.state === "MOVING";
      const isFlashing = trackInfo?.isFlashing;

      const rawColor = getClassColor(det.class_id);
      const color = isFlashing ? "#10b981" : rawColor;
      const confidencePct = `${Math.round((det.confidence || 0) * 100)}%`;

      const trackIdText = trackInfo?.id
        ? `#${String(trackInfo.id).replace(/^trk-/, "")} `
        : det.track_id !== undefined && det.track_id !== null
        ? `#${det.track_id} `
        : "";
      const guardPrefix = isGuarded ? "🛡️ " : "";
      const statusText = isFlashing ? " [Moved ✓]" : isMoving ? " [Moving ➔]" : "";
      const labelText = `${guardPrefix}${trackIdText}${det.label} ${confidencePct}${statusText}`;

      // Semi-transparent fill
      ctx.fillStyle = isFlashing ? "rgba(16, 185, 129, 0.2)" : isMoving ? `${color}28` : `${color}1A`;
      ctx.fillRect(x, y, w, h);

      // Crisp bounding box outline
      ctx.strokeStyle = color;
      ctx.lineWidth = isFlashing || isMoving ? 3.5 : 2.5;
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
      ctx.fillStyle = isFlashing ? "#ffffff" : "#090d16";
      ctx.fillText(labelText, pillX + pillPaddingX, pillY + 4);
    });

    // 2. Draw Missing Items Footprints (Red Dashed Ghost Boxes)
    if (Array.isArray(missingItems) && missingItems.length > 0) {
      missingItems.forEach((item) => {
        if (!Array.isArray(item.lastBbox) || item.lastBbox.length < 4) return;
        const [mx1, my1, mx2, my2] = item.lastBbox;
        const gx = offsetX + mx1 * scaleX;
        const gy = offsetY + my1 * scaleY;
        const gw = Math.max(4, (mx2 - mx1) * scaleX);
        const gh = Math.max(4, (my2 - my1) * scaleY);

        ctx.save();
        ctx.setLineDash([6, 5]);
        ctx.strokeStyle = "#ef4444";
        ctx.lineWidth = 3;
        ctx.strokeRect(gx, gy, gw, gh);
        ctx.fillStyle = "rgba(239, 68, 68, 0.16)";
        ctx.fillRect(gx, gy, gw, gh);

        // Missing item alert pill
        const missingText = `⚠️ MISSING: ${item.label.toUpperCase()}`;
        ctx.font = "700 12px Inter, -apple-system, BlinkMacSystemFont, sans-serif";
        const mm = ctx.measureText(missingText);
        const pillW = mm.width + 16;
        const pillH = 22;
        const pillX = Math.max(offsetX, Math.min(gx, offsetX + renderWidth - pillW));
        const pillY = gy - pillH >= offsetY ? gy - pillH : gy;

        ctx.fillStyle = "#ef4444";
        ctx.beginPath();
        if (typeof ctx.roundRect === "function") {
          ctx.roundRect(pillX, pillY, pillW, pillH, 4);
        } else {
          ctx.rect(pillX, pillY, pillW, pillH);
        }
        ctx.fill();

        ctx.fillStyle = "#ffffff";
        ctx.fillText(missingText, pillX + 8, pillY + 4);

        // Subtext inside box
        ctx.font = "600 10px Inter, -apple-system, BlinkMacSystemFont, sans-serif";
        ctx.fillStyle = "#fca5a5";
        ctx.fillText("Item removed from this spot", gx + 6, gy + 14);

        ctx.restore();
      });
    }

    ctx.restore();
  }, [activeTracksMap, cameraStatus, detections, isGuarded, missingItems, sourceFrameSize, videoRef]);

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
