import React, { useCallback, useEffect, useRef, useState } from "react";
import { FiCheck, FiCrosshair, FiMaximize, FiX } from "react-icons/fi";
import { autoBoxCustomImage } from "../services/api";

const MIN_BOX_SIZE = 0.04;
const FULL_BOX = [0.02, 0.02, 0.98, 0.98];
const DEFAULT_BOX = [0.18, 0.18, 0.82, 0.82];

function clamp(val, min, max) {
  return Math.max(min, Math.min(max, val));
}

function createCroppedPreviewDataUrl(imageUrl, bboxNorm) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      try {
        const w = img.naturalWidth || img.width;
        const h = img.naturalHeight || img.height;
        const [x1, y1, x2, y2] = bboxNorm;
        const sx = Math.max(0, Math.round(x1 * w));
        const sy = Math.max(0, Math.round(y1 * h));
        const sw = Math.max(4, Math.round((x2 - x1) * w));
        const sh = Math.max(4, Math.round((y2 - y1) * h));

        const maxThumb = 180;
        const scale = Math.min(1, maxThumb / Math.max(sw, sh));
        const canvas = document.createElement("canvas");
        canvas.width = Math.max(1, Math.round(sw * scale));
        canvas.height = Math.max(1, Math.round(sh * scale));
        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);
        resolve(canvas.toDataURL("image/jpeg", 0.85));
      } catch {
        resolve(imageUrl);
      }
    };
    img.onerror = () => resolve(imageUrl);
    img.src = imageUrl;
  });
}

export function ObjectBoxModal({ isOpen, items = [], onCancel, onConfirm }) {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [boxes, setBoxes] = useState([]); // Array of [x1, y1, x2, y2]
  const [isAutoDetecting, setIsAutoDetecting] = useState(false);
  const [autoBoxNote, setAutoBoxNote] = useState("");

  const stageRef = useRef(null);
  const dragRef = useRef(null);
  const autoCheckedRef = useRef(new Set());

  // Reset state whenever a new batch of items opens
  useEffect(() => {
    if (!isOpen || !items.length) return;
    setCurrentIndex(0);
    setBoxes(items.map((it) => (it.bboxNorm ? [...it.bboxNorm] : [...DEFAULT_BOX])));
    autoCheckedRef.current = new Set();
    setAutoBoxNote("");
  }, [isOpen, items]);

  const currentItem = items[currentIndex] || null;
  const currentBox = boxes[currentIndex] || DEFAULT_BOX;

  const runAutoBoxForIndex = useCallback(
    async (idx, file) => {
      if (!file) return;
      setIsAutoDetecting(true);
      setAutoBoxNote("Detecting main object...");
      try {
        const res = await autoBoxCustomImage(file);
        if (res && Array.isArray(res.bbox_norm) && res.bbox_norm.length === 4) {
          setBoxes((prev) => {
            const next = [...prev];
            next[idx] = res.bbox_norm;
            return next;
          });
          setAutoBoxNote(
            res.method === "grabcut"
              ? "Auto-boxed foreground — drag corners to fine-tune"
              : "Drag around your object to exclude background"
          );
        }
      } catch {
        setAutoBoxNote("Drag a tight box around your object");
      } finally {
        setIsAutoDetecting(false);
      }
    },
    []
  );

  // Automatically run GrabCut Auto-Box the first time each photo without a preset box is shown
  useEffect(() => {
    if (!isOpen || !currentItem) return;
    if (currentItem.bboxNorm) {
      setAutoBoxNote("Drag corners or redraw to adjust selection");
      return;
    }
    if (autoCheckedRef.current.has(currentIndex)) return;
    autoCheckedRef.current.add(currentIndex);
    runAutoBoxForIndex(currentIndex, currentItem.file);
  }, [currentIndex, currentItem, isOpen, runAutoBoxForIndex]);

  // Escape key cancels modal
  useEffect(() => {
    if (!isOpen) return;
    const onKey = (e) => {
      if (e.key === "Escape") onCancel();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [isOpen, onCancel]);

  const getNormalizedPoint = (clientX, clientY) => {
    const el = stageRef.current;
    if (!el) return null;
    const rect = el.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return null;
    return {
      x: clamp((clientX - rect.left) / rect.width, 0, 1),
      y: clamp((clientY - rect.top) / rect.height, 0, 1),
    };
  };

  const updateCurrentBox = (nextBox) => {
    setBoxes((prev) => {
      const copy = [...prev];
      copy[currentIndex] = nextBox;
      return copy;
    });
  };

  const handleStagePointerDown = (e) => {
    if (e.button !== 0 && e.pointerType === "mouse") return;
    const pt = getNormalizedPoint(e.clientX, e.clientY);
    if (!pt) return;

    e.currentTarget.setPointerCapture(e.pointerId);
    const [x1, y1, x2, y2] = currentBox;

    // If clicking inside existing box, move it; otherwise draw a fresh box
    if (pt.x > x1 && pt.x < x2 && pt.y > y1 && pt.y < y2) {
      dragRef.current = {
        mode: "move",
        startX: pt.x,
        startY: pt.y,
        origBox: [x1, y1, x2, y2],
      };
    } else {
      dragRef.current = {
        mode: "draw",
        startX: pt.x,
        startY: pt.y,
        origBox: [pt.x, pt.y, pt.x, pt.y],
      };
    }
  };

  const handleCornerPointerDown = (corner, e) => {
    e.stopPropagation();
    const pt = getNormalizedPoint(e.clientX, e.clientY);
    if (!pt) return;
    if (stageRef.current) {
      stageRef.current.setPointerCapture(e.pointerId);
    }
    dragRef.current = {
      mode: corner, // 'nw' | 'ne' | 'sw' | 'se'
      startX: pt.x,
      startY: pt.y,
      origBox: [...currentBox],
    };
  };

  const handleStagePointerMove = (e) => {
    const drag = dragRef.current;
    if (!drag) return;
    const pt = getNormalizedPoint(e.clientX, e.clientY);
    if (!pt) return;

    const [ox1, oy1, ox2, oy2] = drag.origBox;

    if (drag.mode === "draw") {
      const nx1 = Math.min(drag.startX, pt.x);
      const ny1 = Math.min(drag.startY, pt.y);
      const nx2 = Math.max(drag.startX, pt.x);
      const ny2 = Math.max(drag.startY, pt.y);
      if (nx2 - nx1 >= 0.02 && ny2 - ny1 >= 0.02) {
        updateCurrentBox([nx1, ny1, nx2, ny2]);
      }
      return;
    }

    if (drag.mode === "move") {
      const bw = ox2 - ox1;
      const bh = oy2 - oy1;
      const dx = pt.x - drag.startX;
      const dy = pt.y - drag.startY;
      const nx1 = clamp(ox1 + dx, 0, 1 - bw);
      const ny1 = clamp(oy1 + dy, 0, 1 - bh);
      updateCurrentBox([nx1, ny1, nx1 + bw, ny1 + bh]);
      return;
    }

    // Corner resize handles
    let nx1 = ox1;
    let ny1 = oy1;
    let nx2 = ox2;
    let ny2 = oy2;

    if (drag.mode.includes("w")) {
      nx1 = clamp(pt.x, 0, ox2 - MIN_BOX_SIZE);
    }
    if (drag.mode.includes("e")) {
      nx2 = clamp(pt.x, ox1 + MIN_BOX_SIZE, 1);
    }
    if (drag.mode.includes("n")) {
      ny1 = clamp(pt.y, 0, oy2 - MIN_BOX_SIZE);
    }
    if (drag.mode.includes("s")) {
      ny2 = clamp(pt.y, oy1 + MIN_BOX_SIZE, 1);
    }

    updateCurrentBox([nx1, ny1, nx2, ny2]);
  };

  const handleStagePointerUp = (e) => {
    if (dragRef.current) {
      dragRef.current = null;
      try {
        e.currentTarget.releasePointerCapture(e.pointerId);
      } catch {
        // Ignore if capture already released
      }
    }
  };

  const handleConfirmStep = async () => {
    if (currentIndex < items.length - 1) {
      setCurrentIndex((prev) => prev + 1);
      return;
    }

    // Final step: build tight cropped preview URLs for every photo and return
    const results = await Promise.all(
      items.map(async (item, idx) => {
        const box = boxes[idx] || DEFAULT_BOX;
        const roundedBox = box.map((v) => Number(v.toFixed(4)));
        const croppedPreviewUrl = await createCroppedPreviewDataUrl(
          item.previewUrl,
          roundedBox
        );
        return {
          file: item.file,
          previewUrl: item.previewUrl,
          croppedPreviewUrl,
          bboxNorm: roundedBox,
        };
      })
    );
    onConfirm(results);
  };

  if (!isOpen || !currentItem) {
    return null;
  }

  const [bx1, by1, bx2, by2] = currentBox;
  const areaPct = Math.round((bx2 - bx1) * (by2 - by1) * 100);
  const isLast = currentIndex === items.length - 1;

  return (
    <div className="modal-backdrop" onClick={onCancel} role="dialog" aria-modal="true">
      <div className="modal-card box-select-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-header-info">
            <div className="modal-title-row">
              <h2>Select Only Your Object</h2>
              {items.length > 1 && (
                <span className="box-step-badge">
                  Photo {currentIndex + 1} of {items.length}
                </span>
              )}
            </div>
            <p className="modal-subtitle">
              Draw a tight box around the item so the background is ignored during detection.
            </p>
          </div>
          <button
            type="button"
            className="modal-close-btn"
            onClick={onCancel}
            aria-label="Cancel selection"
          >
            <FiX />
          </button>
        </div>

        {/* Toolbar */}
        <div className="box-select-toolbar">
          <div className="box-toolbar-left">
            <button
              type="button"
              className="btn-box-tool"
              onClick={() => runAutoBoxForIndex(currentIndex, currentItem.file)}
              disabled={isAutoDetecting}
              title="Automatically locate foreground object"
            >
              <FiCrosshair aria-hidden="true" />
              <span>{isAutoDetecting ? "Auto-Boxing..." : "Auto-Box"}</span>
            </button>
            <button
              type="button"
              className="btn-box-tool"
              onClick={() => updateCurrentBox([...FULL_BOX])}
              disabled={isAutoDetecting}
              title="Select entire photo"
            >
              <FiMaximize aria-hidden="true" />
              <span>Full Image</span>
            </button>
          </div>

          <div className="box-toolbar-right">
            <span className="box-area-pill">
              Selected: <strong>{areaPct}%</strong> of image
            </span>
          </div>
        </div>

        {/* Interactive Image Stage */}
        <div className="box-stage-outer">
          <div
            ref={stageRef}
            className="box-stage-inner"
            onPointerDown={handleStagePointerDown}
            onPointerMove={handleStagePointerMove}
            onPointerUp={handleStagePointerUp}
            onPointerCancel={handleStagePointerUp}
          >
            <img
              src={currentItem.previewUrl}
              alt="Select target object"
              className="box-stage-img"
              draggable={false}
            />

            {/* Draggable / Resizable Selection Box with Spotlight Dim Outside */}
            <div
              className="box-selection-rect"
              style={{
                left: `${bx1 * 100}%`,
                top: `${by1 * 100}%`,
                width: `${(bx2 - bx1) * 100}%`,
                height: `${(by2 - by1) * 100}%`,
              }}
            >
              <span className={`box-selection-tag ${by1 < 0.08 ? "tag-below" : "tag-above"}`}>
                Target Object
              </span>
              <span
                className="box-corner-handle handle-nw"
                onPointerDown={(e) => handleCornerPointerDown("nw", e)}
              />
              <span
                className="box-corner-handle handle-ne"
                onPointerDown={(e) => handleCornerPointerDown("ne", e)}
              />
              <span
                className="box-corner-handle handle-sw"
                onPointerDown={(e) => handleCornerPointerDown("sw", e)}
              />
              <span
                className="box-corner-handle handle-se"
                onPointerDown={(e) => handleCornerPointerDown("se", e)}
              />
            </div>
          </div>
          {autoBoxNote && <p className="box-stage-hint">{autoBoxNote}</p>}
        </div>

        {/* Modal Footer */}
        <div className="modal-footer">
          {currentIndex > 0 ? (
            <button
              type="button"
              className="btn btn-secondary"
              onClick={() => setCurrentIndex((prev) => prev - 1)}
            >
              &larr; Previous
            </button>
          ) : (
            <button type="button" className="btn btn-secondary" onClick={onCancel}>
              Cancel
            </button>
          )}

          <button
            type="button"
            className="btn btn-primary"
            onClick={handleConfirmStep}
            disabled={isAutoDetecting}
          >
            <FiCheck style={{ marginRight: 6 }} />
            {isLast ? "Confirm Selection" : `Next Photo (${currentIndex + 2}/${items.length})`}
          </button>
        </div>
      </div>
    </div>
  );
}

