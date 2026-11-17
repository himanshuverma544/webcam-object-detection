import React, { useState } from "react";
import { FiCheck, FiSearch, FiX } from "react-icons/fi";
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

export function DetectionList({
  supportedClasses = [],
  selectedClasses = [],
  onSelectClasses,
  detectionCount = 0,
}) {
  const [searchQuery, setSearchQuery] = useState("");

  const allClasses =
    Array.isArray(supportedClasses) && supportedClasses.length > 0
      ? supportedClasses
      : DEFAULT_COCO_CLASSES;

  // Split classes into Active (selected) and Inactive (unselected)
  const activeClasses = allClasses.filter((cls) => selectedClasses.includes(cls));
  const inactiveClasses = allClasses.filter((cls) => !selectedClasses.includes(cls));

  // Filter both groups in-place based on the user's search query
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

  return (
    <aside className="detection-sidebar-card" aria-label="Supported COCO classes selector">
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
    </aside>
  );
}
