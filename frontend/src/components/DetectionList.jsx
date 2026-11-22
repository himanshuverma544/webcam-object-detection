import React, { useEffect, useRef, useState } from "react";
import {
  FiCamera,
  FiCheck,
  FiCrop,
  FiImage,
  FiPlus,
  FiSearch,
  FiTrash2,
  FiUpload,
  FiX,
} from "react-icons/fi";
import { ObjectBoxModal } from "./ObjectBoxModal";
import { getClassColor } from "./VideoViewport";

const DEFAULT_COCO_CLASSES = [
  "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
  "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
  "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
  "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
  "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
  "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
  "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
  "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
  "scissors", "teddy bear", "hair drier", "toothbrush",
];

const BACKEND_SUPPORTED_TYPES = new Set([
  "image/jpeg",
  "image/jpg",
  "image/png",
  "image/webp",
]);

function normalizeImageFile(file) {
  if (file.type && BACKEND_SUPPORTED_TYPES.has(file.type.toLowerCase())) {
    return Promise.resolve(file);
  }
  return new Promise((resolve) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      try {
        const canvas = document.createElement("canvas");
        canvas.width = img.naturalWidth || img.width;
        canvas.height = img.naturalHeight || img.height;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0);
        canvas.toBlob(
          (blob) => {
            URL.revokeObjectURL(url);
            if (blob) {
              const baseName = (file.name || "photo").replace(/\.[^.]+$/, "");
              resolve(new File([blob], `${baseName}.jpg`, { type: "image/jpeg" }));
            } else {
              resolve(file);
            }
          },
          "image/jpeg",
          0.9
        );
      } catch {
        URL.revokeObjectURL(url);
        resolve(file);
      }
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      resolve(file);
    };
    img.src = url;
  });
}

export const DetectionList = React.forwardRef(function DetectionList(
  {
    supportedClasses = [],
    selectedClasses = [],
    onSelectClasses,
    customItems = [],
    selectedCustomIds = [],
    onToggleCustomItem,
    onUploadCustomItem,
    onAddCustomItemImages,
    onDeleteCustomItem,
    similarityThreshold = 0.65,
    onSimilarityChange,
    isUploadingCustom = false,
    customItemError = null,
    onClearCustomError,
    cameraStatus = "idle",
    onCaptureLiveSnapshot,
  },
  ref
) {
  const [activeTab, setActiveTab] = useState("coco"); // 'coco' | 'custom'
  const [searchQuery, setSearchQuery] = useState("");

  // Custom item upload form state
  const [itemName, setItemName] = useState("");
  const [stagedPhotos, setStagedPhotos] = useState([]); // [{ file, previewUrl, croppedPreviewUrl, bboxNorm }]
  const [localFormError, setLocalFormError] = useState(null);
  const [boxModalState, setBoxModalState] = useState(null);
  const nameInputRef = useRef(null);
  const fileInputRef = useRef(null);
  const addAngleInputRef = useRef(null);
  const addAngleTargetIdRef = useRef(null);

  React.useImperativeHandle(ref, () => ({
    stageCopilotCapture: async ({ name, file, onComplete, onCancel }) => {
      setActiveTab("custom");
      setLocalFormError(null);
      if (onClearCustomError) onClearCustomError(null);

      const normalized = await normalizeImageFile(file);
      const previewUrl = URL.createObjectURL(normalized);

      setBoxModalState({
        mode: "copilotStage",
        name,
        items: [{ file: normalized, previewUrl }],
        onComplete,
        onCancel,
      });
    },
    isCropModalOpen: () => Boolean(boxModalState),
  }));

  // Revoke object URLs on unmount
  useEffect(() => {
    return () => {
      stagedPhotos.forEach((p) => URL.revokeObjectURL(p.previewUrl));
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const allClasses =
    Array.isArray(supportedClasses) && supportedClasses.length > 0
      ? supportedClasses
      : DEFAULT_COCO_CLASSES;

  // Split COCO classes into Active (selected) and Inactive (unselected)
  const activeClasses = allClasses.filter((cls) => selectedClasses.includes(cls));
  const inactiveClasses = allClasses.filter((cls) => !selectedClasses.includes(cls));

  const trimmedQuery = searchQuery.trim().toLowerCase();
  const visibleActive = trimmedQuery
    ? activeClasses.filter((cls) => cls.toLowerCase().includes(trimmedQuery))
    : activeClasses;
  const visibleInactive = trimmedQuery
    ? inactiveClasses.filter((cls) => cls.toLowerCase().includes(trimmedQuery))
    : inactiveClasses;

  const totalVisible = visibleActive.length + visibleInactive.length;

  const handleToggleClass = (clsName) => {
    if (!onSelectClasses) return;
    if (selectedClasses.includes(clsName)) {
      onSelectClasses(selectedClasses.filter((c) => c !== clsName));
    } else {
      onSelectClasses([...selectedClasses, clsName]);
    }
  };

  const handleClearAll = () => {
    if (onSelectClasses) {
      onSelectClasses([]);
    }
  };

  const handleSelectAll = () => {
    if (onSelectClasses) {
      onSelectClasses([...allClasses]);
    }
  };

  const stageFiles = async (fileList) => {
    const rawFiles = Array.from(fileList || []).filter((f) =>
      f.type ? f.type.startsWith("image/") : true
    );
    if (rawFiles.length === 0) {
      setLocalFormError("Please select valid image files (JPEG, PNG, WebP, etc.).");
      return;
    }

    setLocalFormError(null);
    if (onClearCustomError) onClearCustomError(null);

    const normalizedFiles = await Promise.all(rawFiles.map(normalizeImageFile));
    const prepared = normalizedFiles.map((file) => ({
      file,
      previewUrl: URL.createObjectURL(file),
    }));

    // Open interactive box selector with Auto-Box Assist so background is cleanly excluded
    setBoxModalState({
      mode: "stage",
      items: prepared,
    });
  };

  const handleEditStagedBox = (idx) => {
    const target = stagedPhotos[idx];
    if (!target) return;
    setBoxModalState({
      mode: "edit",
      editIndex: idx,
      items: [
        {
          file: target.file,
          previewUrl: target.previewUrl,
          bboxNorm: target.bboxNorm,
        },
      ],
    });
  };

  const handleBoxModalCancel = () => {
    if (boxModalState) {
      if (boxModalState.mode === "copilotStage" && typeof boxModalState.onCancel === "function") {
        boxModalState.onCancel(boxModalState.name);
      }
      if (boxModalState.mode !== "edit") {
        boxModalState.items.forEach((it) => URL.revokeObjectURL(it.previewUrl));
      }
    }
    setBoxModalState(null);
  };

  const handleBoxModalConfirm = async (results) => {
    const currentModal = boxModalState;
    setBoxModalState(null);
    if (!currentModal || !Array.isArray(results) || results.length === 0) return;

    if (currentModal.mode === "copilotStage") {
      const targetName = currentModal.name || itemName || "Custom Item";
      try {
        await onUploadCustomItem(
          targetName,
          results.map((r) => r.file),
          results.map((r) => r.bboxNorm)
        );
        setItemName("");
        setStagedPhotos([]);
        if (typeof currentModal.onComplete === "function") {
          currentModal.onComplete(targetName);
        }
      } catch (err) {
        // Error surfaced via customItemError
      } finally {
        results.forEach((r) => URL.revokeObjectURL(r.previewUrl));
      }
      return;
    }

    if (currentModal.mode === "stage") {
      setStagedPhotos((prev) => [...prev, ...results]);
      return;
    }

    if (currentModal.mode === "edit" && typeof currentModal.editIndex === "number") {
      const updated = results[0];
      setStagedPhotos((prev) =>
        prev.map((p, idx) => (idx === currentModal.editIndex ? updated : p))
      );
      return;
    }

    if (currentModal.mode === "addAngle" && currentModal.targetItemId && onAddCustomItemImages) {
      try {
        await onAddCustomItemImages(
          currentModal.targetItemId,
          results.map((r) => r.file),
          results.map((r) => r.bboxNorm)
        );
      } catch {
        // Error surfaced via customItemError
      } finally {
        results.forEach((r) => URL.revokeObjectURL(r.previewUrl));
      }
    }
  };

  const handleRemoveStaged = (idx) => {
    setStagedPhotos((prev) => {
      const target = prev[idx];
      if (target?.previewUrl) {
        URL.revokeObjectURL(target.previewUrl);
      }
      return prev.filter((_, i) => i !== idx);
    });
  };

  const handleSnapFromCamera = async () => {
    if (!onCaptureLiveSnapshot) return;
    setLocalFormError(null);
    if (onClearCustomError) onClearCustomError(null);
    try {
      const snapFile = await onCaptureLiveSnapshot();
      stageFiles([snapFile]);
    } catch (err) {
      setLocalFormError(err.message || "Could not capture snapshot from camera.");
    }
  };

  const handleCreateCustomItem = async (e) => {
    e.preventDefault();
    const cleaned = itemName.trim();
    if (!cleaned) {
      setLocalFormError("Please enter a name for your item (e.g. My Office Chair).");
      nameInputRef.current?.focus();
      return;
    }
    const isDuplicate = customItems.some(
      (it) => it.name.trim().toLowerCase() === cleaned.toLowerCase()
    );
    if (isDuplicate) {
      setLocalFormError(`A custom item named '${cleaned}' already exists. Please choose a unique name.`);
      nameInputRef.current?.focus();
      return;
    }
    if (stagedPhotos.length === 0) {
      setLocalFormError("Please upload at least 1 reference photo of your item.");
      fileInputRef.current?.click();
      return;
    }
    setLocalFormError(null);
    if (onClearCustomError) onClearCustomError(null);

    try {
      await onUploadCustomItem(
        cleaned,
        stagedPhotos.map((p) => p.file),
        stagedPhotos.map((p) => p.bboxNorm)
      );
      stagedPhotos.forEach((p) => URL.revokeObjectURL(p.previewUrl));
      setStagedPhotos([]);
      setItemName("");
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    } catch {
      // Error surfaced via customItemError
    }
  };

  const triggerAddAngle = (itemId, e) => {
    e.stopPropagation();
    addAngleTargetIdRef.current = itemId;
    if (addAngleInputRef.current) {
      addAngleInputRef.current.value = "";
      addAngleInputRef.current.click();
    }
  };

  const handleAddAngleSelected = async (e) => {
    const targetId = addAngleTargetIdRef.current;
    const rawFiles = Array.from(e.target.files || []);
    e.target.value = "";
    if (!targetId || rawFiles.length === 0 || !onAddCustomItemImages) return;

    const normalized = await Promise.all(rawFiles.map(normalizeImageFile));
    const prepared = normalized.map((file) => ({
      file,
      previewUrl: URL.createObjectURL(file),
    }));
    setBoxModalState({
      mode: "addAngle",
      targetItemId: targetId,
      items: prepared,
    });
  };

  const activeCustomCount = selectedCustomIds.length;
  const displayedError = localFormError || customItemError;

  return (
    <aside className="detection-sidebar-card" aria-label="Object detection target selector">
      {/* Mode Tabs: COCO Classes vs Custom Reference Items */}
      <div className="sidebar-tabs" role="tablist" aria-label="Detection target tabs">
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === "coco"}
          className={`sidebar-tab-btn ${activeTab === "coco" ? "tab-active" : ""}`}
          onClick={() => setActiveTab("coco")}
        >
          <span>COCO Classes</span>
          <span className={`tab-count-pill ${activeClasses.length > 0 ? "pill-lit" : ""}`}>
            {activeClasses.length}/{allClasses.length}
          </span>
        </button>

        <button
          type="button"
          role="tab"
          aria-selected={activeTab === "custom"}
          className={`sidebar-tab-btn ${activeTab === "custom" ? "tab-active" : ""}`}
          onClick={() => setActiveTab("custom")}
        >
          <span>My Items</span>
          <span className={`tab-count-pill ${activeCustomCount > 0 ? "pill-lit" : ""}`}>
            {activeCustomCount > 0 ? `${activeCustomCount}/${customItems.length}` : customItems.length}
          </span>
        </button>
      </div>

      {activeTab === "coco" ? (
        <>
          <div className="sidebar-header">
            <div>
              <h2>Supported COCO Classes</h2>
              <p className="sidebar-subtitle">
                Click pills to select or deselect which objects to detect
              </p>
            </div>
            <span
              className={`count-badge ${activeClasses.length > 0 ? "has-selection" : ""}`}
              title={`${activeClasses.length} of ${allClasses.length} classes active`}
            >
              {activeClasses.length} / {allClasses.length}
            </span>
          </div>

          {/* Toolbar with Select All and Clear All displayed using justify-between end to end */}
          <div className="classes-toolbar-row">
            <button
              type="button"
              className="btn-classes-toggle-all btn-select-all"
              onClick={handleSelectAll}
              title="Select all 80 classes"
              disabled={activeClasses.length === allClasses.length}
            >
              Select All
            </button>
            <button
              type="button"
              className="btn-classes-toggle-all btn-clear-all"
              onClick={handleClearAll}
              title="Clear all selected classes"
              disabled={activeClasses.length === 0}
            >
              Clear All
            </button>
          </div>

          {/* In-place search field to filter pills */}
          <div className="classes-search-bar">
            <div className="search-input-wrapper">
              <FiSearch className="search-field-icon" aria-hidden="true" />
              <input
                type="text"
                className="classes-filter-input"
                placeholder="Search classes (e.g. car, cup, phone)..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                aria-label="Filter COCO classes pills"
              />
              {searchQuery && (
                <button
                  type="button"
                  className="btn-search-clear"
                  onClick={() => setSearchQuery("")}
                  title="Clear search filter"
                  aria-label="Clear search query"
                >
                  <FiX />
                </button>
              )}
            </div>
            {trimmedQuery && (
              <div className="search-results-meta">
                <span>
                  Showing {totalVisible} of {allClasses.length} classes
                </span>
              </div>
            )}
          </div>

          {/* Scrollable area with two dividers: Active on top, Inactive below */}
          <div className="classes-pill-scroll-area">
            {trimmedQuery && totalVisible === 0 ? (
              <div className="no-pills-found">
                <p>No COCO classes match &ldquo;{searchQuery}&rdquo;</p>
                <button
                  type="button"
                  className="btn-reset-search"
                  onClick={() => setSearchQuery("")}
                >
                  Show all 80 classes
                </button>
              </div>
            ) : (
              <>
                {/* 1. Active Divider */}
                <div className="classes-section-divider active-divider" aria-label="Active classes divider">
                  <span className="divider-line" />
                  <span className="divider-badge active-badge">
                    Active ({activeClasses.length})
                  </span>
                  <span className="divider-line" />
                </div>

                {/* Active Pills */}
                {visibleActive.length === 0 ? (
                  <p className="empty-group-note">
                    {trimmedQuery
                      ? `No active classes match "${searchQuery}"`
                      : "No active classes. Click pills below to select."}
                  </p>
                ) : (
                  <div
                    className="classes-pill-container active-group"
                    role="group"
                    aria-label="Active COCO classes"
                  >
                    {visibleActive.map((clsName) => {
                      const classIdx = allClasses.indexOf(clsName);
                      const color = getClassColor(classIdx >= 0 ? classIdx : 0);
                      return (
                        <button
                          key={clsName}
                          type="button"
                          className="coco-class-pill selected"
                          onClick={() => handleToggleClass(clsName)}
                          title={`Click to deselect '${clsName}'`}
                          style={{
                            borderColor: color,
                            backgroundColor: "rgba(56, 189, 248, 0.16)",
                            color: "#ffffff",
                          }}
                          aria-pressed="true"
                        >
                          <FiCheck className="pill-check-icon" />
                          <span>{clsName}</span>
                        </button>
                      );
                    })}
                  </div>
                )}

                {/* 2. Inactive Divider */}
                <div className="classes-section-divider inactive-divider" aria-label="Inactive classes divider">
                  <span className="divider-line" />
                  <span className="divider-badge inactive-badge">
                    Inactive ({inactiveClasses.length})
                  </span>
                  <span className="divider-line" />
                </div>

                {/* Inactive Pills */}
                {visibleInactive.length === 0 ? (
                  <p className="empty-group-note">
                    {trimmedQuery
                      ? `No inactive classes match "${searchQuery}"`
                      : "All classes are currently active."}
                  </p>
                ) : (
                  <div
                    className="classes-pill-container inactive-group"
                    role="group"
                    aria-label="Inactive COCO classes"
                  >
                    {visibleInactive.map((clsName) => (
                      <button
                        key={clsName}
                        type="button"
                        className="coco-class-pill"
                        onClick={() => handleToggleClass(clsName)}
                        title={`Click to detect '${clsName}'`}
                        aria-pressed="false"
                      >
                        <span>{clsName}</span>
                      </button>
                    ))}
                  </div>
                )}
              </>
            )}
          </div>
        </>
      ) : (
        <div className="custom-items-panel">
          <div className="sidebar-header">
            <div>
              <h2>My Custom Items</h2>
              <p className="sidebar-subtitle">
                Upload photos of your specific chair or any personal item to detect only that item
              </p>
            </div>
          </div>

          {/* Add Custom Item Form */}
          <form className="custom-upload-form" onSubmit={handleCreateCustomItem}>
            <input
              ref={nameInputRef}
              type="text"
              className="custom-name-input"
              placeholder="Item name (e.g. My Office Chair, Keys)..."
              value={itemName}
              maxLength={60}
              onChange={(e) => {
                setItemName(e.target.value);
                if (localFormError) setLocalFormError(null);
              }}
              disabled={isUploadingCustom}
              aria-label="Custom item name"
            />

            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              multiple
              className="sr-only-file-input"
              onChange={(e) => {
                const picked = Array.from(e.target.files || []);
                e.target.value = "";
                if (picked.length > 0) {
                  stageFiles(picked);
                }
              }}
            />

            <input
              ref={addAngleInputRef}
              type="file"
              accept="image/*"
              multiple
              className="sr-only-file-input"
              onChange={handleAddAngleSelected}
            />

            <div className="custom-upload-actions">
              <button
                type="button"
                className="btn-custom-action"
                onClick={() => fileInputRef.current?.click()}
                disabled={isUploadingCustom}
                title="Upload reference photos of your specific item"
              >
                <FiUpload aria-hidden="true" />
                <span>Upload Photos</span>
              </button>

              <button
                type="button"
                className="btn-custom-action"
                onClick={handleSnapFromCamera}
                disabled={isUploadingCustom || cameraStatus !== "active"}
                title={
                  cameraStatus === "active"
                    ? "Capture current webcam frame as reference photo"
                    : "Start camera to capture a live reference photo"
                }
              >
                <FiCamera aria-hidden="true" />
                <span>Snap Camera</span>
              </button>
            </div>

            {stagedPhotos.length > 0 && (
              <div className="staged-thumbs-row">
                {stagedPhotos.map((p, idx) => (
                  <div
                    key={p.previewUrl}
                    className="staged-thumb-item"
                    title="Click to adjust object selection box"
                    onClick={() => handleEditStagedBox(idx)}
                  >
                    <img
                      src={p.croppedPreviewUrl || p.previewUrl}
                      alt={`Selected object ${idx + 1}`}
                    />
                    <span className="staged-thumb-crop-badge" aria-hidden="true">
                      <FiCrop />
                    </span>
                    <button
                      type="button"
                      className="staged-thumb-remove"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleRemoveStaged(idx);
                      }}
                      aria-label={`Remove photo ${idx + 1}`}
                    >
                      <FiX />
                    </button>
                  </div>
                ))}
                <button
                  type="button"
                  className="staged-thumb-add-btn"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isUploadingCustom}
                  title="Add more reference photos"
                  aria-label="Add more reference photos"
                >
                  <FiPlus />
                </button>
              </div>
            )}

            {displayedError && (
              <div className="custom-form-error" role="alert">
                {displayedError}
              </div>
            )}

            <button
              type="submit"
              className="btn-custom-submit"
              disabled={isUploadingCustom || stagedPhotos.length === 0}
            >
              {isUploadingCustom
                ? "Indexing Visual Prompt..."
                : stagedPhotos.length > 0
                ? `Save & Detect Item (${stagedPhotos.length} ${stagedPhotos.length === 1 ? "photo" : "photos"})`
                : "Save & Detect Item"}
            </button>
          </form>

          {/* Match Strictness Slider */}
          <div className="custom-strictness-box">
            <div className="slider-label-row">
              <label htmlFor="similarity-slider">Match Strictness</label>
              <span className="slider-value">
                {Math.round(similarityThreshold * 100)}%
              </span>
            </div>
            <input
              id="similarity-slider"
              type="range"
              min="0.40"
              max="0.90"
              step="0.01"
              value={similarityThreshold}
              onChange={(e) =>
                onSimilarityChange && onSimilarityChange(parseFloat(e.target.value))
              }
            />
            <span className="strictness-hint">
              Higher rejects other chairs; lower helps in dim lighting
            </span>
          </div>

          {/* Saved Custom Items List */}
          <div className="custom-items-scroll-area">
            {customItems.length === 0 ? (
              <div className="custom-empty-state">
                <FiImage className="custom-empty-icon" aria-hidden="true" />
                <p>No custom items saved yet.</p>
                <span>
                  Upload 1–3 photos of your chair or any object above to track only your specific item.
                </span>
              </div>
            ) : (
              <div className="custom-items-list">
                {customItems.map((item) => {
                  const isActive = selectedCustomIds.includes(item.id);
                  const accent = getClassColor(item.class_id);
                  return (
                    <div
                      key={item.id}
                      className={`custom-item-card ${isActive ? "is-active" : ""}`}
                      style={isActive ? { borderColor: accent } : undefined}
                      onClick={() => onToggleCustomItem && onToggleCustomItem(item.id)}
                      role="button"
                      tabIndex={0}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" || e.key === " ") {
                          e.preventDefault();
                          if (onToggleCustomItem) onToggleCustomItem(item.id);
                        }
                      }}
                      aria-pressed={isActive}
                    >
                      <div className="custom-item-thumb">
                        {item.thumbnail_base64 ? (
                          <img src={item.thumbnail_base64} alt={item.name} />
                        ) : (
                          <FiImage />
                        )}
                      </div>

                      <div className="custom-item-meta">
                        <div className="custom-item-name-row">
                          <span className="custom-item-name">{item.name}</span>
                          {isActive && (
                            <span
                              className="custom-active-chip"
                              style={{ borderColor: accent, color: accent }}
                            >
                              <FiCheck /> Active
                            </span>
                          )}
                        </div>
                        <span className="custom-item-sub">
                          {item.image_count} reference photo{item.image_count === 1 ? "" : "s"}
                        </span>
                      </div>

                      <div className="custom-item-card-actions">
                        <button
                          type="button"
                          className="btn-custom-icon"
                          onClick={(e) => triggerAddAngle(item.id, e)}
                          disabled={isUploadingCustom}
                          title="Add another angle photo to improve accuracy"
                          aria-label={`Add angle photo for ${item.name}`}
                        >
                          <FiPlus />
                        </button>
                        <button
                          type="button"
                          className="btn-custom-icon btn-custom-delete"
                          onClick={(e) => {
                            e.stopPropagation();
                            if (onDeleteCustomItem) onDeleteCustomItem(item.id);
                          }}
                          title={`Delete ${item.name}`}
                          aria-label={`Delete ${item.name}`}
                        >
                          <FiTrash2 />
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      )}
      <ObjectBoxModal
        isOpen={Boolean(boxModalState)}
        items={boxModalState?.items || []}
        onCancel={handleBoxModalCancel}
        onConfirm={handleBoxModalConfirm}
      />
    </aside>
  );
});
