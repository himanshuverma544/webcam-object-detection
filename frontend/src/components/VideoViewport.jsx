import React, { useCallback, useEffect, useRef, useState } from "react";

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

const DEFAULT_ZOOM = { scale: 1, tx: 0, ty: 0 };

export function getClassColor(classId = 0) {
  const index = Math.abs(Number(classId) || 0) % BOX_PALETTE.length;
  return BOX_PALETTE[index];
}

function getBboxCenter(bbox) {
  if (!Array.isArray(bbox) || bbox.length < 4) return null;
  return {
    x: (bbox[0] + bbox[2]) / 2,
    y: (bbox[1] + bbox[3]) / 2,
  };
}

function isSameDetection(det, target) {
  if (!det || !target) return false;
  if (det.class_id !== target.classId) return false;

  if (
    det.track_id !== null &&
    det.track_id !== undefined &&
    target.trackId !== null &&
    target.trackId !== undefined
  ) {
    return det.track_id === target.trackId;
  }

  const c1 = getBboxCenter(det.bbox);
  const c2 = getBboxCenter(target.bbox);
  if (!c1 || !c2) return false;
  return Math.hypot(c1.x - c2.x, c1.y - c2.y) < 35;
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
  const stageRef = useRef(null);
  const overlayCanvasRef = useRef(null);

  const [focusedTarget, setFocusedTarget] = useState(null);
  const [zoomTransform, setZoomTransform] = useState(DEFAULT_ZOOM);
  const [isHoveringBox, setIsHoveringBox] = useState(false);

  const zoomTransformRef = useRef(DEFAULT_ZOOM);
  const focusedTargetRef = useRef(null);

  useEffect(() => {
    zoomTransformRef.current = zoomTransform;
  }, [zoomTransform]);

  useEffect(() => {
    focusedTargetRef.current = focusedTarget;
  }, [focusedTarget]);

  const resetZoom = useCallback(() => {
    focusedTargetRef.current = null;
    setFocusedTarget(null);
    setZoomTransform(DEFAULT_ZOOM);
  }, []);

  // Compute unscaled layout metrics of the rendered video content inside .video-stage
  const getLayoutMetrics = useCallback(() => {
    const video = videoRef.current;
    const stage = stageRef.current;
    if (!video || !stage) return null;

    // clientWidth/clientHeight are unaffected by CSS transform: scale(...)
    const displayWidth = Math.round(video.clientWidth || stage.clientWidth || 0);
    const displayHeight = Math.round(video.clientHeight || stage.clientHeight || 0);

    if (displayWidth <= 0 || displayHeight <= 0) {
      return null;
    }

    const srcWidth = sourceFrameSize?.width || video.videoWidth || displayWidth;
    const srcHeight = sourceFrameSize?.height || video.videoHeight || displayHeight;

    if (srcWidth <= 0 || srcHeight <= 0) {
      return null;
    }

    // Actual rendered video content box inside the <video> element (object-fit: contain)
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

    return {
      displayWidth,
      displayHeight,
      srcWidth,
      srcHeight,
      renderWidth,
      renderHeight,
      offsetX,
      offsetY,
      scaleX: renderWidth / srcWidth,
      scaleY: renderHeight / srcHeight,
    };
  }, [sourceFrameSize, videoRef]);

  // Compute clamped translate + scale (with transform-origin: 0 0) to center & focus on bbox
  const computeZoomForBbox = useCallback((bbox, layout, prevTransform = null) => {
    if (!Array.isArray(bbox) || bbox.length < 4 || !layout) {
      return DEFAULT_ZOOM;
    }

    const [rawX1, rawY1, rawX2, rawY2] = bbox;
    const bx1 = layout.offsetX + rawX1 * layout.scaleX;
    const by1 = layout.offsetY + rawY1 * layout.scaleY;
    const bx2 = layout.offsetX + rawX2 * layout.scaleX;
    const by2 = layout.offsetY + rawY2 * layout.scaleY;

    const bw = Math.max(24, bx2 - bx1);
    const bh = Math.max(24, by2 - by1);
    const cx = (bx1 + bx2) / 2;
    const cy = (by1 + by2) / 2;

    // Fit object comfortably at ~55% of viewport, clamped between 1.5x and 3.5x
    const targetFillRatio = 0.55;
    const idealScale = Math.min(
      (layout.displayWidth * targetFillRatio) / bw,
      (layout.displayHeight * targetFillRatio) / bh
    );
    const clampedScale = Math.max(1.5, Math.min(3.5, idealScale));

    // Gently damp scale changes during live auto-tracking to avoid "breathing" jitter
    const scale =
      prevTransform && prevTransform.scale > 1
        ? prevTransform.scale * 0.72 + clampedScale * 0.28
        : clampedScale;

    // Clamp translation so zoomed video never exposes empty void outside edges
    const minTx = layout.displayWidth * (1 - scale);
    const minTy = layout.displayHeight * (1 - scale);
    const tx = Math.max(minTx, Math.min(0, layout.displayWidth / 2 - cx * scale));
    const ty = Math.max(minTy, Math.min(0, layout.displayHeight / 2 - cy * scale));

    // Deadzone filter during tracking to prevent micro-shake from minor bbox flicker
    if (
      prevTransform &&
      prevTransform.scale > 1 &&
      Math.abs(tx - prevTransform.tx) < 6 &&
      Math.abs(ty - prevTransform.ty) < 6 &&
      Math.abs(scale - prevTransform.scale) < 0.08
    ) {
      return prevTransform;
    }

    return {
      scale: Number(scale.toFixed(3)),
      tx: Math.round(tx),
      ty: Math.round(ty),
    };
  }, []);

  // Hit-test pointer coordinates against detected bounding boxes (smallest area wins on overlap)
  const findDetectionAtClientPoint = useCallback(
    (clientX, clientY) => {
      if (
        cameraStatus !== "active" ||
        !Array.isArray(detections) ||
        detections.length === 0
      ) {
        return null;
      }

      const canvas = overlayCanvasRef.current;
      const layout = getLayoutMetrics();
      if (!canvas || !layout) return null;

      // canvas.getBoundingClientRect() accurately reflects instantaneous CSS transform even mid-transition
      const canvasRect = canvas.getBoundingClientRect();
      if (canvasRect.width <= 0 || canvasRect.height <= 0) return null;

      const lx = ((clientX - canvasRect.left) / canvasRect.width) * layout.displayWidth;
      const ly = ((clientY - canvasRect.top) / canvasRect.height) * layout.displayHeight;

      const uiScale = 1 / Math.max(1, zoomTransformRef.current.scale);
      const hitPad = 5 * uiScale;
      const pillPad = 24 * uiScale;

      let bestMatch = null;
      let smallestArea = Infinity;

      for (const det of detections) {
        if (!Array.isArray(det.bbox) || det.bbox.length < 4) continue;
        const [rawX1, rawY1, rawX2, rawY2] = det.bbox;
        const x1 = layout.offsetX + rawX1 * layout.scaleX;
        const y1 = layout.offsetY + rawY1 * layout.scaleY;
        const x2 = layout.offsetX + rawX2 * layout.scaleX;
        const y2 = layout.offsetY + rawY2 * layout.scaleY;

        if (
          lx >= x1 - hitPad &&
          lx <= x2 + hitPad &&
          ly >= y1 - pillPad &&
          ly <= y2 + hitPad
        ) {
          const area = Math.max(1, (x2 - x1) * (y2 - y1));
          if (area < smallestArea) {
            smallestArea = area;
            bestMatch = det;
          }
        }
      }

      return bestMatch;
    },
    [cameraStatus, detections, getLayoutMetrics]
  );

  const handleCanvasClick = useCallback(
    (e) => {
      if (cameraStatus !== "active") return;
      const hit = findDetectionAtClientPoint(e.clientX, e.clientY);

      if (!hit) {
        if (focusedTarget) {
          resetZoom();
        }
        return;
      }

      // Clicking the already-focused object toggles zoom back out
      if (focusedTarget && isSameDetection(hit, focusedTarget)) {
        resetZoom();
        return;
      }

      const layout = getLayoutMetrics();
      if (!layout) return;

      const nextTarget = {
        trackId: hit.track_id ?? null,
        classId: hit.class_id,
        label: hit.label,
        bbox: [...hit.bbox],
        lost: false,
      };
      const nextZoom = computeZoomForBbox(hit.bbox, layout, null);
      focusedTargetRef.current = nextTarget;
      setFocusedTarget(nextTarget);
      setZoomTransform(nextZoom);
    },
    [cameraStatus, computeZoomForBbox, findDetectionAtClientPoint, focusedTarget, getLayoutMetrics, resetZoom]
  );

  const handleCanvasMouseMove = useCallback(
    (e) => {
      const hit = findDetectionAtClientPoint(e.clientX, e.clientY);
      setIsHoveringBox(Boolean(hit));
    },
    [findDetectionAtClientPoint]
  );

  // Reset zoom automatically when camera stops
  useEffect(() => {
    if (cameraStatus !== "active") {
      focusedTargetRef.current = null;
      setFocusedTarget(null);
      setZoomTransform(DEFAULT_ZOOM);
      setIsHoveringBox(false);
    }
  }, [cameraStatus]);

  // Escape key exits zoom focus
  useEffect(() => {
    if (!focusedTarget) return;
    const handleKeyDown = (e) => {
      if (e.key === "Escape") {
        resetZoom();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [focusedTarget, resetZoom]);

  // Live auto-follow: track focused object across detection frames
  useEffect(() => {
    const currentTarget = focusedTargetRef.current;
    if (!currentTarget || cameraStatus !== "active") return;

    const layout = getLayoutMetrics();
    if (!layout) return;

    let match = null;
    if (Array.isArray(detections) && detections.length > 0) {
      // 1. Prefer exact track_id match
      if (currentTarget.trackId !== null && currentTarget.trackId !== undefined) {
        match = detections.find(
          (d) =>
            d.track_id === currentTarget.trackId &&
            d.class_id === currentTarget.classId
        );
      }

      // 2. Fallback: closest detection of same class within reasonable range
      if (!match) {
        const lastCenter = getBboxCenter(currentTarget.bbox);
        const maxJump = Math.max(layout.srcWidth, layout.srcHeight) * 0.28;
        let closestDist = maxJump;
        for (const d of detections) {
          if (d.class_id !== currentTarget.classId) continue;
          const c = getBboxCenter(d.bbox);
          if (!c || !lastCenter) continue;
          const dist = Math.hypot(c.x - lastCenter.x, c.y - lastCenter.y);
          if (dist < closestDist) {
            closestDist = dist;
            match = d;
          }
        }
      }
    }

    if (match) {
      setFocusedTarget((prev) => {
        if (!prev) return null;
        const updated = {
          ...prev,
          trackId: match.track_id ?? prev.trackId,
          bbox: [...match.bbox],
          lost: false,
        };
        focusedTargetRef.current = updated;
        return updated;
      });
      const nextZoom = computeZoomForBbox(match.bbox, layout, zoomTransformRef.current);
      setZoomTransform((prev) =>
        prev.scale === nextZoom.scale &&
        prev.tx === nextZoom.tx &&
        prev.ty === nextZoom.ty
          ? prev
          : nextZoom
      );
    } else {
      setFocusedTarget((prev) => {
        if (!prev || prev.lost) return prev;
        const updated = { ...prev, lost: true };
        focusedTargetRef.current = updated;
        return updated;
      });
    }
  }, [cameraStatus, computeZoomForBbox, detections, getLayoutMetrics]);

  const drawOverlay = useCallback(() => {
    const canvas = overlayCanvasRef.current;
    const layout = getLayoutMetrics();
    if (!canvas || !layout) return;

    const {
      displayWidth,
      displayHeight,
      renderWidth,
      offsetX,
      offsetY,
      scaleX,
      scaleY,
    } = layout;

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

    if (cameraStatus !== "active") {
      ctx.restore();
      return;
    }

    const hasDetections = Array.isArray(detections) && detections.length > 0;
    const hasMissing = Array.isArray(missingItems) && missingItems.length > 0;

    if (!hasDetections && !hasMissing && !focusedTarget) {
      ctx.restore();
      return;
    }

    // Scale UI strokes and pills inversely to zoom so boxes stay crisp instead of chunky
    const currentZoomScale = Math.max(1, zoomTransform.scale);
    const uiScale = 1 / currentZoomScale;
    const fontScale = 1 / Math.pow(currentZoomScale, 0.72);

    // 1. Draw Ghost Origin Boxes and Trajectory Lines for moving or recently moved items
    if (activeTracksMap && typeof activeTracksMap === "object") {
      ctx.save();
      if (focusedTarget) {
        ctx.globalAlpha = 0.35;
      }
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
          ctx.setLineDash([5 * uiScale, 4 * uiScale]);
          ctx.strokeStyle = track.isFlashing ? "#10b98188" : `${color}99`;
          ctx.lineWidth = 2 * uiScale;
          ctx.strokeRect(gx, gy, gw, gh);
          ctx.fillStyle = `${color}0D`;
          ctx.fillRect(gx, gy, gw, gh);

          // Small origin indicator label
          const originLabel = "Origin";
          ctx.font = `600 ${Math.max(7, Math.round(10 * fontScale))}px Inter, -apple-system, BlinkMacSystemFont, sans-serif`;
          ctx.textBaseline = "top";
          const om = ctx.measureText(originLabel);
          const opW = om.width + 8 * fontScale;
          const opH = 16 * fontScale;
          ctx.fillStyle = track.isFlashing ? "#10b981" : `${color}CC`;
          if (typeof ctx.roundRect === "function") {
            ctx.roundRect(gx, Math.max(offsetY, gy - opH), opW, opH, 3 * uiScale);
          } else {
            ctx.rect(gx, Math.max(offsetY, gy - opH), opW, opH);
          }
          ctx.fill();
          ctx.fillStyle = "#ffffff";
          ctx.fillText(originLabel, gx + 4 * fontScale, Math.max(offsetY, gy - opH) + 2 * fontScale);
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
          ctx.lineWidth = (track.isFlashing ? 3.5 : 2.5) * uiScale;
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
            ctx.arc(px, py, (idx === path.length - 1 ? 4 : 2.5) * uiScale, 0, Math.PI * 2);
            ctx.fill();
          });

          // Draw Arrowhead at destination
          const last = path[path.length - 1];
          const prev = path[Math.max(0, path.length - 2)];
          const angle = Math.atan2((last.y - prev.y) * scaleY, (last.x - prev.x) * scaleX);
          const arrowLength = 12 * uiScale;
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
            const distText = track.isFlashing ? `Moved ${track.distance}px` : `${track.distance}px`;
            ctx.font = `600 ${Math.max(7, Math.round(11 * fontScale))}px Inter, -apple-system, BlinkMacSystemFont, sans-serif`;
            ctx.textBaseline = "top";
            const dm = ctx.measureText(distText);
            const badgeW = dm.width + 10 * fontScale;
            const badgeH = 18 * fontScale;

            ctx.fillStyle = "rgba(9, 13, 22, 0.85)";
            if (typeof ctx.roundRect === "function") {
              ctx.roundRect(midX - badgeW / 2, midY - badgeH / 2, badgeW, badgeH, 4 * uiScale);
            } else {
              ctx.rect(midX - badgeW / 2, midY - badgeH / 2, badgeW, badgeH);
            }
            ctx.fill();
            ctx.strokeStyle = track.isFlashing ? "#10b981" : color;
            ctx.lineWidth = 1 * uiScale;
            ctx.stroke();

            ctx.fillStyle = track.isFlashing ? "#34d399" : "#ffffff";
            ctx.fillText(distText, midX - badgeW / 2 + 5 * fontScale, midY - badgeH / 2 + 2 * fontScale);
          }

          ctx.restore();
        }
      });
      ctx.restore();
    }

    // Helper to render a single detection box
    const drawDetectionBox = (det, isFocused, dimUnfocused) => {
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
      const scoreText =
        det.is_custom && typeof det.similarity === "number"
          ? `${Math.round(det.similarity * 100)}% match`
          : confidencePct;

      const trackIdText = trackInfo?.id
        ? `#${String(trackInfo.id).replace(/^trk-/, "")} `
        : det.track_id !== undefined && det.track_id !== null
        ? `#${det.track_id} `
        : "";
      const guardPrefix = isGuarded ? "[GUARD] " : "";
      const customPrefix = det.is_custom ? "[MY ITEM] " : "";
      const statusText = isFlashing ? " [Moved]" : isMoving ? " [Moving]" : "";
      const labelText = `${guardPrefix}${customPrefix}${trackIdText}${det.label} ${scoreText}${statusText}`;

      ctx.save();
      if (dimUnfocused) {
        ctx.globalAlpha = 0.35;
      }

      // Semi-transparent fill (omit fill on focused object so the zoomed subject stays crystal clear)
      if (!isFocused) {
        ctx.fillStyle = isFlashing
          ? "rgba(16, 185, 129, 0.2)"
          : isMoving
          ? `${color}28`
          : `${color}1A`;
        ctx.fillRect(x, y, w, h);
      }

      // Bounding box outline
      ctx.strokeStyle = isFocused ? `${color}CC` : color;
      ctx.lineWidth = (isFocused ? 2 : isFlashing || isMoving ? 3.5 : 2.5) * uiScale;
      ctx.strokeRect(x, y, w, h);

      // Focus reticle corner brackets for the zoomed/focused object
      if (isFocused) {
        const cornerLen = Math.min(w * 0.22, h * 0.22, 18 * uiScale);
        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth = 3.5 * uiScale;
        ctx.lineCap = "round";
        ctx.shadowColor = color;
        ctx.shadowBlur = 10 * uiScale;

        ctx.beginPath();
        // Top-left
        ctx.moveTo(x, y + cornerLen);
        ctx.lineTo(x, y);
        ctx.lineTo(x + cornerLen, y);
        // Top-right
        ctx.moveTo(x + w - cornerLen, y);
        ctx.lineTo(x + w, y);
        ctx.lineTo(x + w, y + cornerLen);
        // Bottom-right
        ctx.moveTo(x + w, y + h - cornerLen);
        ctx.lineTo(x + w, y + h);
        ctx.lineTo(x + w - cornerLen, y + h);
        // Bottom-left
        ctx.moveTo(x + cornerLen, y + h);
        ctx.lineTo(x, y + h);
        ctx.lineTo(x, y + h - cornerLen);
        ctx.stroke();
        ctx.restore();
      }

      // Label pill
      const fontSize = Math.max(8, Math.round(13 * fontScale));
      ctx.font = `600 ${fontSize}px Inter, -apple-system, BlinkMacSystemFont, sans-serif`;
      ctx.textBaseline = "top";
      const textMetrics = ctx.measureText(labelText);
      const pillPaddingX = 8 * fontScale;
      const pillHeight = 22 * fontScale;
      const pillWidth = Math.min(renderWidth, textMetrics.width + pillPaddingX * 2);

      const pillX = Math.max(offsetX, Math.min(x, offsetX + renderWidth - pillWidth));
      const pillY = y - pillHeight >= offsetY ? y - pillHeight : y;

      ctx.fillStyle = color;
      ctx.beginPath();
      if (typeof ctx.roundRect === "function") {
        ctx.roundRect(pillX, pillY, pillWidth, pillHeight, 4 * uiScale);
      } else {
        ctx.rect(pillX, pillY, pillWidth, pillHeight);
      }
      ctx.fill();

      ctx.fillStyle = isFlashing ? "#ffffff" : "#090d16";
      ctx.fillText(labelText, pillX + pillPaddingX, pillY + 4 * fontScale);
      ctx.restore();
    };

    // Separate unfocused vs focused detections so spotlight dimming sits cleanly in between
    let matchedFocusedDet = null;
    const unfocusedDets = [];

    if (hasDetections) {
      detections.forEach((det) => {
        if (focusedTarget && !matchedFocusedDet && isSameDetection(det, focusedTarget)) {
          matchedFocusedDet = det;
        } else {
          unfocusedDets.push(det);
        }
      });
    }

    // Draw unfocused detections first
    unfocusedDets.forEach((det) => drawDetectionBox(det, false, Boolean(focusedTarget)));

    // Draw spotlight dimming mask around focused object
    if (focusedTarget) {
      const focusBbox = matchedFocusedDet?.bbox || focusedTarget.bbox;
      if (Array.isArray(focusBbox) && focusBbox.length >= 4) {
        const [fx1, fy1, fx2, fy2] = focusBbox;
        const pad = 6 * uiScale;
        const cutX = offsetX + fx1 * scaleX - pad;
        const cutY = offsetY + fy1 * scaleY - pad;
        const cutW = Math.max(4, (fx2 - fx1) * scaleX) + pad * 2;
        const cutH = Math.max(4, (fy2 - fy1) * scaleY) + pad * 2;

        ctx.save();
        ctx.fillStyle = "rgba(6, 9, 17, 0.52)";
        ctx.beginPath();
        ctx.rect(0, 0, displayWidth, displayHeight);
        ctx.rect(cutX, cutY, cutW, cutH);
        ctx.fill("evenodd");
        ctx.restore();
      }

      // Draw focused detection brightly on top of the spotlight
      if (matchedFocusedDet) {
        drawDetectionBox(matchedFocusedDet, true, false);
      } else if (Array.isArray(focusedTarget.bbox) && focusedTarget.bbox.length >= 4) {
        // Draw dashed last-known reticle while target is temporarily lost
        const [lx1, ly1, lx2, ly2] = focusedTarget.bbox;
        const rx = offsetX + lx1 * scaleX;
        const ry = offsetY + ly1 * scaleY;
        const rw = Math.max(4, (lx2 - lx1) * scaleX);
        const rh = Math.max(4, (ly2 - ly1) * scaleY);
        ctx.save();
        ctx.setLineDash([6 * uiScale, 5 * uiScale]);
        ctx.strokeStyle = "#fbbf24";
        ctx.lineWidth = 2.5 * uiScale;
        ctx.strokeRect(rx, ry, rw, rh);
        ctx.restore();
      }
    }

    // 2. Draw Missing Items Footprints (Red Dashed Ghost Boxes)
    if (hasMissing) {
      missingItems.forEach((item) => {
        if (!Array.isArray(item.lastBbox) || item.lastBbox.length < 4) return;
        const [mx1, my1, mx2, my2] = item.lastBbox;
        const gx = offsetX + mx1 * scaleX;
        const gy = offsetY + my1 * scaleY;
        const gw = Math.max(4, (mx2 - mx1) * scaleX);
        const gh = Math.max(4, (my2 - my1) * scaleY);

        ctx.save();
        ctx.setLineDash([6 * uiScale, 5 * uiScale]);
        ctx.strokeStyle = "#ef4444";
        ctx.lineWidth = 3 * uiScale;
        ctx.strokeRect(gx, gy, gw, gh);
        ctx.fillStyle = "rgba(239, 68, 68, 0.16)";
        ctx.fillRect(gx, gy, gw, gh);

        // Missing item alert pill
        const missingText = `MISSING: ${item.label.toUpperCase()}`;
        ctx.font = `700 ${Math.max(8, Math.round(12 * fontScale))}px Inter, -apple-system, BlinkMacSystemFont, sans-serif`;
        ctx.textBaseline = "top";
        const mm = ctx.measureText(missingText);
        const pillW = mm.width + 16 * fontScale;
        const pillH = 22 * fontScale;
        const pillX = Math.max(offsetX, Math.min(gx, offsetX + renderWidth - pillW));
        const pillY = gy - pillH >= offsetY ? gy - pillH : gy;

        ctx.fillStyle = "#ef4444";
        ctx.beginPath();
        if (typeof ctx.roundRect === "function") {
          ctx.roundRect(pillX, pillY, pillW, pillH, 4 * uiScale);
        } else {
          ctx.rect(pillX, pillY, pillW, pillH);
        }
        ctx.fill();

        ctx.fillStyle = "#ffffff";
        ctx.fillText(missingText, pillX + 8 * fontScale, pillY + 4 * fontScale);

        // Subtext inside box
        ctx.font = `600 ${Math.max(7, Math.round(10 * fontScale))}px Inter, -apple-system, BlinkMacSystemFont, sans-serif`;
        ctx.fillStyle = "#fca5a5";
        ctx.fillText("Item removed from this spot", gx + 6 * fontScale, gy + 14 * fontScale);

        ctx.restore();
      });
    }

    ctx.restore();
  }, [
    activeTracksMap,
    cameraStatus,
    detections,
    focusedTarget,
    getLayoutMetrics,
    isGuarded,
    missingItems,
    zoomTransform.scale,
  ]);

  // Redraw whenever detections, focus state, or frame dimensions update
  useEffect(() => {
    drawOverlay();
  }, [drawOverlay]);

  // Keep bounding boxes and zoom aligned when the video container or window resizes
  useEffect(() => {
    const stageEl = stageRef.current;
    const containerEl = containerRef.current;

    const handleResize = () => {
      drawOverlay();
      if (focusedTarget?.bbox) {
        const layout = getLayoutMetrics();
        if (layout) {
          setZoomTransform(computeZoomForBbox(focusedTarget.bbox, layout, null));
        }
      }
    };

    window.addEventListener("resize", handleResize);

    let resizeObserver;
    if (typeof ResizeObserver !== "undefined") {
      resizeObserver = new ResizeObserver(() => {
        handleResize();
      });
      if (stageEl) resizeObserver.observe(stageEl);
      if (containerEl) resizeObserver.observe(containerEl);
    }

    return () => {
      window.removeEventListener("resize", handleResize);
      if (resizeObserver) {
        resizeObserver.disconnect();
      }
    };
  }, [computeZoomForBbox, drawOverlay, focusedTarget, getLayoutMetrics]);

  return (
    <div className="video-viewport-card" ref={containerRef}>
      <div className="video-stage" ref={stageRef}>
        <div
          className={`video-zoom-layer ${focusedTarget ? "is-zoomed" : ""}`}
          style={{
            transform: `translate3d(${zoomTransform.tx}px, ${zoomTransform.ty}px, 0) scale(${zoomTransform.scale})`,
          }}
        >
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
            className={`bbox-overlay-canvas ${
              cameraStatus === "active" ? "canvas-interactive" : ""
            } ${isHoveringBox ? "canvas-hover-box" : ""}`}
            onClick={handleCanvasClick}
            onMouseMove={handleCanvasMouseMove}
            onMouseLeave={() => setIsHoveringBox(false)}
            aria-label="Object detection bounding box overlay"
          />
        </div>

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
