import React, { useCallback, useEffect, useRef, useState } from "react";
import { ChatCopilot } from "./components/ChatCopilot";
import { ControlPanel } from "./components/ControlPanel";
import { DetectionList } from "./components/DetectionList";
import { Header } from "./components/Header";
import { MetricsModal } from "./components/MetricsModal";
import { MissingAlertBanner } from "./components/MissingAlertBanner";
import { VideoViewport } from "./components/VideoViewport";
import { useSceneTracker } from "./hooks/useSceneTracker";
import { useSessionMetrics } from "./hooks/useSessionMetrics";
import { useWebcamDetection } from "./hooks/useWebcamDetection";
import { API_BASE_URL, resetTrack, sendMetricsEmail } from "./services/api";
import { generateExecutiveSummary } from "./utils/summaryGenerator";
import "./App.css";

export function App() {
  const {
    videoRef,
    cameraStatus,
    cameraError,
    videoDimensions,
    isDetecting,
    detectionStatus,
    detectionError,
    confidence,
    setConfidence,
    captureIntervalMs,
    setCaptureIntervalMs,
    detections,
    detectionCount,
    inferenceTimeMs,
    roundTripTimeMs,
    sourceFrameSize,
    lastUpdatedAt,
    backendStatus,
    healthInfo,
    supportedClasses,
    selectedClasses,
    setSelectedClasses,
    customItems,
    selectedCustomIds,
    similarityThreshold,
    setSimilarityThreshold,
    isUploadingCustom,
    customItemError,
    setCustomItemError,
    uploadCustomItem,
    addImagesToCustomItem,
    removeCustomItem,
    toggleCustomItem,
    refreshCustomItems,
    captureLiveSnapshotFile,
    startCamera,
    stopCamera,
    toggleDetection,
    checkBackendHealth,
    handleVideoMetadataLoaded,
  } = useWebcamDetection();

  const activeMonitoredNames = [
    ...selectedClasses,
    ...customItems
      .filter((item) => selectedCustomIds.includes(item.id))
      .map((item) => item.name),
  ];

  const {
    activeTracksMap,
    activeMovingCount,
    movementEvents,
    missingItems,
    missingEvents,
    isGuarded,
    guardedCount,
    minMoveDistance,
    setMinMoveDistance,
    settleDwellTimeMs,
    setSettleDwellTimeMs,
    toggleGuard,
    dismissMissingItem,
    clearEvents,
    resetScene,
    processFrame,
  } = useSceneTracker();

  const [isMetricsModalOpen, setIsMetricsModalOpen] = useState(false);
  const [metricsModalTab, setMetricsModalTab] = useState("metrics");
  const [copiedSummaryTick, setCopiedSummaryTick] = useState(0);
  const [emailNotice, setEmailNotice] = useState(null);
  const detectionListRef = useRef(null);

  // Auto-dismiss successful email banner after 6 seconds
  useEffect(() => {
    if (emailNotice && emailNotice.type === "success") {
      const timer = setTimeout(() => {
        setEmailNotice(null);
      }, 6000);
      return () => clearTimeout(timer);
    }
  }, [emailNotice]);

  // Session metrics accumulator
  const {
    completedMetrics,
    hasMetrics,
    latestMetricsRef,
    snapshotSession,
  } = useSessionMetrics({
    cameraStatus,
    isDetecting,
    isGuarded,
    detections,
    detectionCount,
    inferenceTimeMs,
    roundTripTimeMs,
    confidence,
    missingEvents,
    missingItems,
    movementEvents,
    activeTracksMap,
    lastUpdatedAt,
    selectedClasses: activeMonitoredNames,
  });

  // Intercept camera stop to freeze session metrics before scene resets
  const handleStopCamera = useCallback(() => {
    const snap = snapshotSession();
    stopCamera();
    return snap;
  }, [snapshotSession, stopCamera]);

  // Run scene tracking once on every completed detection frame
  useEffect(() => {
    if (cameraStatus === "active" && isDetecting) {
      processFrame(detections);
    }
  }, [lastUpdatedAt]); // eslint-disable-line react-hooks/exhaustive-deps

  // Reset scene tracks when camera stops
  useEffect(() => {
    if (cameraStatus !== "active") {
      resetScene();
    }
  }, [cameraStatus, resetScene]);

  // Execute structured AI Copilot action commands on the UI
  const handleExecuteCopilotActions = useCallback(
    (actions, options = {}) => {
      if (!Array.isArray(actions)) return;
      const { onNotifyCopilot } = options;

      // Keep track of synchronously created metrics snapshot (e.g. when stopping camera and sending email in same turn)
      let activeMetricsSnapshot = completedMetrics || latestMetricsRef.current;

      for (const action of actions) {
        if (!action || !action.type) continue;
        const payload = action.payload || {};

        switch (action.type) {
          case "START_CAMERA":
            startCamera();
            break;

          case "STOP_CAMERA": {
            const snap = handleStopCamera();
            if (snap) {
              activeMetricsSnapshot = snap;
            }
            break;
          }

          case "TOGGLE_DETECTION": {
            const act = payload.action;
            if (act === "pause" && isDetecting) {
              toggleDetection();
            } else if (act === "resume" && !isDetecting) {
              toggleDetection();
            } else if (!act || act === "toggle") {
              toggleDetection();
            }
            break;
          }

          case "SET_CONFIDENCE":
            if (typeof payload.value === "number") {
              setConfidence(payload.value);
            }
            break;

          case "SET_FRAME_CADENCE":
            if (typeof payload.interval_ms === "number") {
              setCaptureIntervalMs(payload.interval_ms);
            }
            break;

          case "SET_MIN_MOVE_DISTANCE":
            if (typeof payload.distance_px === "number") {
              setMinMoveDistance(payload.distance_px);
            }
            break;

          case "SET_SETTLE_DWELL_TIME":
            if (typeof payload.dwell_ms === "number") {
              setSettleDwellTimeMs(payload.dwell_ms);
            }
            break;

          case "SET_COCO_CLASSES": {
            const { mode = "replace", classes: targetClasses = [] } = payload;
            if (mode === "select_all") {
              setSelectedClasses(
                supportedClasses.length > 0 ? [...supportedClasses] : [...targetClasses]
              );
            } else if (mode === "clear_all") {
              setSelectedClasses([]);
            } else if (mode === "add") {
              setSelectedClasses((prev) => Array.from(new Set([...prev, ...targetClasses])));
            } else if (mode === "remove") {
              setSelectedClasses((prev) => prev.filter((c) => !targetClasses.includes(c)));
            } else {
              setSelectedClasses([...targetClasses]);
            }
            break;
          }

          case "TOGGLE_CUSTOM_ITEM": {
            const itemRef = String(payload.item_ref || "").toLowerCase();
            const enable = payload.enable;
            const target = customItems.find(
              (it) => it.id === itemRef || it.name.toLowerCase() === itemRef
            );
            if (target) {
              if (enable === true && !selectedCustomIds.includes(target.id)) {
                toggleCustomItem(target.id);
              } else if (enable === false && selectedCustomIds.includes(target.id)) {
                toggleCustomItem(target.id);
              } else if (enable === undefined) {
                toggleCustomItem(target.id);
              }
            }
            break;
          }

          case "DELETE_CUSTOM_ITEM": {
            const itemRef = String(payload.item_ref || "").trim().toLowerCase();
            const target = customItems.find(
              (it) => it.id.toLowerCase() === itemRef || it.name.trim().toLowerCase() === itemRef
            );
            if (target) {
              removeCustomItem(target.id);
              if (typeof onNotifyCopilot === "function") {
                onNotifyCopilot(`🗑️ **Deleted!** Custom item **'${target.name}'** has been removed.`);
              }
            } else {
              if (typeof onNotifyCopilot === "function") {
                onNotifyCopilot(`⚠️ **Notice:** Custom item '${payload.item_ref}' was not found.`);
              }
            }
            break;
          }

          case "ADD_CUSTOM_ITEM_IMAGES": {
            const itemRef = String(payload.item_ref || "").trim().toLowerCase();
            const target = customItems.find(
              (it) => it.id.toLowerCase() === itemRef || it.name.trim().toLowerCase() === itemRef
            );
            if (!target) {
              if (typeof onNotifyCopilot === "function") {
                onNotifyCopilot(
                  `⚠️ **Notice:** Custom item '${payload.item_ref}' was not found. Please create it first.`
                );
              }
              break;
            }

            const filesToUpload =
              Array.isArray(options?.attachedFiles) && options.attachedFiles.length > 0
                ? options.attachedFiles
                : options?.attachedFile
                ? [options.attachedFile]
                : null;

            if (filesToUpload && filesToUpload.length > 0) {
              addImagesToCustomItem(target.id, filesToUpload, null);
              if (typeof onNotifyCopilot === "function") {
                const count = filesToUpload.length;
                onNotifyCopilot(
                  `✅ **Updated!** Added ${count} new reference photo${
                    count > 1 ? "s" : ""
                  } to **'${target.name}'**.`
                );
              }
            } else {
              if (typeof onNotifyCopilot === "function") {
                onNotifyCopilot(
                  `📸 Please take a snapshot or upload photos first, then send "Add these photos to '${target.name}'".`
                );
              }
            }
            break;
          }

          case "SET_SIMILARITY_THRESHOLD":
            if (typeof payload.value === "number") {
              setSimilarityThreshold(payload.value);
              if (typeof onNotifyCopilot === "function") {
                const pct = Math.round(payload.value * 100);
                onNotifyCopilot(`🎯 Match strictness set to **${pct}%**.`);
              }
            }
            break;

          case "CAPTURE_CUSTOM_ITEM": {
            const rawName = payload.name || "Custom Item";
            const itemName = rawName.trim();

            // Duplicate check (case-insensitive)
            const isDuplicate = customItems.some(
              (it) => it.name.trim().toLowerCase() === itemName.toLowerCase()
            );
            if (isDuplicate) {
              if (typeof onNotifyCopilot === "function") {
                onNotifyCopilot(
                  `⚠️ **Notice:** A custom item named '**${itemName}**' already exists. Please choose a unique name or delete the existing item first.`
                );
              }
              break;
            }

            // If reference photos were attached from Copilot (already cropped by user upon uploading/snapping):
            const filesToUpload =
              Array.isArray(options?.attachedFiles) && options.attachedFiles.length > 0
                ? options.attachedFiles
                : options?.attachedFile
                ? [options.attachedFile]
                : null;

            if (filesToUpload && filesToUpload.length > 0) {
              uploadCustomItem(itemName, filesToUpload, null);
              if (typeof onNotifyCopilot === "function") {
                const count = filesToUpload.length;
                onNotifyCopilot(
                  `✅ **Created!** Custom reference item **'${itemName}'** has been saved with ${count} reference photo${count > 1 ? "s" : ""} and enabled for detection.`
                );
              }
              break;
            }

            if (cameraStatus !== "active") {
              if (typeof onNotifyCopilot === "function") {
                onNotifyCopilot(
                  `⚠️ **Notice:** The camera is currently stopped. Please start the camera first so I can capture a snapshot of your item.`
                );
              }
              break;
            }

            if (detectionListRef.current?.isCropModalOpen()) {
              if (typeof onNotifyCopilot === "function") {
                onNotifyCopilot(
                  `⚠️ **Notice:** A crop dialog is already open. Please complete or cancel the current selection first.`
                );
              }
              break;
            }

            captureLiveSnapshotFile(itemName)
              .then((file) => {
                if (!file) {
                  if (typeof onNotifyCopilot === "function") {
                    onNotifyCopilot(`⚠️ **Notice:** Failed to capture live snapshot from camera.`);
                  }
                  return;
                }

                if (detectionListRef.current) {
                  detectionListRef.current.stageCopilotCapture({
                    name: itemName,
                    file,
                    onComplete: (savedName) => {
                      if (typeof onNotifyCopilot === "function") {
                        onNotifyCopilot(
                          `✅ **Created!** Custom reference item **'${savedName}'** has been saved with your custom crop and enabled for detection.`
                        );
                      }
                    },
                    onCancel: (cancelledName) => {
                      if (typeof onNotifyCopilot === "function") {
                        onNotifyCopilot(
                          `ℹ️ Custom item creation for **'${cancelledName}'** was cancelled.`
                        );
                      }
                    },
                  });
                }
              })
              .catch((err) => {
                if (typeof onNotifyCopilot === "function") {
                  onNotifyCopilot(
                    `⚠️ **Error:** Could not capture frame: ${err.message || "Unknown error"}`
                  );
                }
              });
            break;
          }

          case "CREATE_CUSTOM_ITEM":
            refreshCustomItems();
            break;

          case "GUARD_SCENE": {
            const shouldArm = payload.arm;
            if (shouldArm !== undefined && shouldArm !== isGuarded) {
              toggleGuard();
            }
            break;
          }

          case "DISMISS_MISSING_ALERT": {
            const label = payload.label;
            if (label) {
              dismissMissingItem(label);
            }
            break;
          }

          case "SHOW_METRICS": {
            const requestedTab =
              payload.tab === "summary" || payload.view === "summary"
                ? "summary"
                : "metrics";
            setMetricsModalTab(requestedTab);

            let metricsToShow = activeMetricsSnapshot || latestMetricsRef.current || completedMetrics;
            if (!metricsToShow && cameraStatus === "active") {
              metricsToShow = handleStopCamera();
              if (metricsToShow) {
                activeMetricsSnapshot = metricsToShow;
              }
            }
            if (metricsToShow || hasMetrics || cameraStatus !== "active") {
              setIsMetricsModalOpen(true);
            }
            break;
          }

          case "COPY_SUMMARY": {
            let metricsToCopy = activeMetricsSnapshot || latestMetricsRef.current || completedMetrics;
            if (!metricsToCopy && cameraStatus === "active") {
              metricsToCopy = handleStopCamera();
              if (metricsToCopy) {
                activeMetricsSnapshot = metricsToCopy;
              }
            }

            if (metricsToCopy) {
              const summaryContent = generateExecutiveSummary(metricsToCopy);
              if (navigator?.clipboard?.writeText) {
                navigator.clipboard.writeText(summaryContent).catch(() => {
                  const textarea = document.createElement("textarea");
                  textarea.value = summaryContent;
                  document.body.appendChild(textarea);
                  textarea.select();
                  try {
                    document.execCommand("copy");
                  } catch (e) {}
                  document.body.removeChild(textarea);
                });
              } else {
                const textarea = document.createElement("textarea");
                textarea.value = summaryContent;
                document.body.appendChild(textarea);
                textarea.select();
                try {
                  document.execCommand("copy");
                } catch (e) {}
                document.body.removeChild(textarea);
              }
              setCopiedSummaryTick((t) => t + 1);
              if (typeof onNotifyCopilot === "function") {
                onNotifyCopilot("📋 **Copied!** Executive session summary has been copied to your clipboard.");
              }
            } else {
              if (typeof onNotifyCopilot === "function") {
                onNotifyCopilot("⚠️ **Notice:** No session metrics recorded yet. Start and stop the camera first to generate a summary.");
              }
            }
            break;
          }

          case "SEND_METRICS_EMAIL": {
            let metricsToSend = activeMetricsSnapshot || latestMetricsRef.current || completedMetrics;
            if (!metricsToSend && cameraStatus === "active") {
              metricsToSend = handleStopCamera();
              if (metricsToSend) {
                activeMetricsSnapshot = metricsToSend;
              }
            }

            if (metricsToSend) {
              const targetEmail = (payload.recipient_email || "").trim() || undefined;
              setEmailNotice({
                type: "info",
                message: targetEmail
                  ? `Dispatching metrics report to ${targetEmail}...`
                  : "Dispatching metrics report via email...",
              });
              sendMetricsEmail(metricsToSend, targetEmail)
                .then((res) => {
                  const successMsg =
                    res?.message ||
                    `Metrics report successfully sent${res?.recipient ? ` to ${res.recipient}` : ""}!`;
                  setEmailNotice({
                    type: "success",
                    message: successMsg,
                  });
                  if (typeof onNotifyCopilot === "function") {
                    onNotifyCopilot(`📧 **Email Sent:** ${successMsg}`);
                  }
                })
                .catch((err) => {
                  const errorMsg =
                    err?.message ||
                    "Failed to dispatch metrics report email. Check backend settings.";
                  setEmailNotice({
                    type: "error",
                    message: errorMsg,
                  });
                  if (typeof onNotifyCopilot === "function") {
                    onNotifyCopilot(`❌ **Email Failed:** ${errorMsg}`);
                  }
                });
            } else {
              const warnMsg =
                "No session metrics recorded yet. Start and stop the camera first to generate session metrics.";
              setEmailNotice({
                type: "warning",
                message: warnMsg,
              });
              if (typeof onNotifyCopilot === "function") {
                onNotifyCopilot(`⚠️ **Notice:** ${warnMsg}`);
              }
            }
            break;
          }

          case "CLOSE_METRICS_MODAL":
            setIsMetricsModalOpen(false);
            break;

          case "RESET_TRACKER":
            resetScene();
            resetTrack();
            break;

          default:
            break;
        }
      }
    },
    [
      startCamera,
      handleStopCamera,
      isDetecting,
      toggleDetection,
      setConfidence,
      setCaptureIntervalMs,
      setMinMoveDistance,
      setSettleDwellTimeMs,
      supportedClasses,
      setSelectedClasses,
      customItems,
      selectedCustomIds,
      toggleCustomItem,
      removeCustomItem,
      setSimilarityThreshold,
      cameraStatus,
      captureLiveSnapshotFile,
      uploadCustomItem,
      refreshCustomItems,
      isGuarded,
      toggleGuard,
      dismissMissingItem,
      completedMetrics,
      hasMetrics,
      latestMetricsRef,
      resetScene,
    ]
  );

  return (
    <div className="app-shell">
      <Header
        backendStatus={backendStatus}
        healthInfo={healthInfo}
        detectionStatus={detectionStatus}
        onRetryHealth={checkBackendHealth}
      />

      {detectionError && (
        <div className="alert-banner alert-warning" role="status">
          <strong>Detection Notice:</strong> {detectionError}
        </div>
      )}

      {backendStatus === "disconnected" && !detectionError && (
        <div className="alert-banner alert-error" role="alert">
          <div>
            <strong>Backend Offline:</strong> Unable to connect to backend at{" "}
            <code>{API_BASE_URL}</code>.
          </div>
          {API_BASE_URL.includes("127.0.0.1") || API_BASE_URL.includes("localhost") ? (
            <p style={{ margin: "6px 0 0", fontSize: "0.82rem", lineHeight: "1.4" }}>
              <strong>Note - How to fix:</strong> Your deployed frontend is still configured to connect to <code>localhost:8000</code>.
              In your Render Frontend dashboard, go to <strong>Environment</strong>, add{" "}
              <code>VITE_API_BASE_URL = https://&lt;your-backend&gt;.onrender.com</code>, then click{" "}
              <strong>Manual Deploy -&gt; Clear build cache &amp; deploy</strong>.
            </p>
          ) : (
            <p style={{ margin: "6px 0 0", fontSize: "0.82rem", lineHeight: "1.4" }}>
              Note: If using Render Free tier, the backend spins down after 15 minutes of inactivity and may take 30–60 seconds to wake up.
            </p>
          )}
        </div>
      )}

      <MissingAlertBanner
        missingItems={missingItems}
        onDismiss={dismissMissingItem}
      />

      {emailNotice && (
        <div
          className={`alert-banner alert-${emailNotice.type}`}
          role="status"
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginTop: "12px",
          }}
        >
          <span>
            <strong>
              {emailNotice.type === "error"
                ? "Email Error:"
                : emailNotice.type === "success"
                ? "Email Sent:"
                : emailNotice.type === "info"
                ? "Email Dispatching:"
                : "Email Notice:"}
            </strong>{" "}
            {emailNotice.message}
          </span>
          <button
            type="button"
            onClick={() => setEmailNotice(null)}
            style={{
              background: "none",
              border: "none",
              color: "inherit",
              cursor: "pointer",
              fontSize: "1.2rem",
              padding: "0 6px",
              lineHeight: 1,
            }}
            title="Dismiss notice"
          >
            ×
          </button>
        </div>
      )}

      <main className="dashboard-grid">
        <div className="primary-column">
          <VideoViewport
            videoRef={videoRef}
            cameraStatus={cameraStatus}
            cameraError={cameraError}
            isDetecting={isDetecting}
            detections={detections}
            sourceFrameSize={sourceFrameSize}
            activeTracksMap={activeTracksMap}
            missingItems={missingItems}
            isGuarded={isGuarded}
            onStartCamera={startCamera}
            onMetadataLoaded={handleVideoMetadataLoaded}
          />

          <ControlPanel
            cameraStatus={cameraStatus}
            isDetecting={isDetecting}
            confidence={confidence}
            onConfidenceChange={setConfidence}
            captureIntervalMs={captureIntervalMs}
            onIntervalChange={setCaptureIntervalMs}
            minMoveDistance={minMoveDistance}
            onMinMoveDistanceChange={setMinMoveDistance}
            settleDwellTimeMs={settleDwellTimeMs}
            onSettleDwellTimeMsChange={setSettleDwellTimeMs}
            activeMovingCount={activeMovingCount}
            movementEventsCount={movementEvents.length}
            isGuarded={isGuarded}
            guardedCount={guardedCount}
            onToggleGuard={toggleGuard}
            missingCount={missingItems.length}
            detectionCount={detectionCount}
            inferenceTimeMs={inferenceTimeMs}
            roundTripTimeMs={roundTripTimeMs}
            videoDimensions={videoDimensions}
            hasMetrics={hasMetrics || Boolean(latestMetricsRef.current)}
            onShowMetrics={() => {
              setMetricsModalTab("metrics");
              setIsMetricsModalOpen(true);
            }}
            onStartCamera={startCamera}
            onStopCamera={handleStopCamera}
            onToggleDetection={toggleDetection}
          />
        </div>

        <div className="secondary-column">
          <DetectionList
            ref={detectionListRef}
            supportedClasses={supportedClasses}
            selectedClasses={selectedClasses}
            onSelectClasses={setSelectedClasses}
            detectionCount={detectionCount}
            customItems={customItems}
            selectedCustomIds={selectedCustomIds}
            onToggleCustomItem={toggleCustomItem}
            onUploadCustomItem={uploadCustomItem}
            onAddCustomItemImages={addImagesToCustomItem}
            onDeleteCustomItem={removeCustomItem}
            similarityThreshold={similarityThreshold}
            onSimilarityChange={setSimilarityThreshold}
            isUploadingCustom={isUploadingCustom}
            customItemError={customItemError}
            onClearCustomError={setCustomItemError}
            cameraStatus={cameraStatus}
            onCaptureLiveSnapshot={captureLiveSnapshotFile}
          />
        </div>
      </main>

      <MetricsModal
        isOpen={isMetricsModalOpen}
        onClose={() => setIsMetricsModalOpen(false)}
        metrics={completedMetrics || latestMetricsRef.current}
        activeTab={metricsModalTab}
        isCopiedExternal={copiedSummaryTick}
      />

      <ChatCopilot
        cameraStatus={cameraStatus}
        isDetecting={isDetecting}
        confidence={confidence}
        similarityThreshold={similarityThreshold}
        selectedClasses={selectedClasses}
        customItems={customItems}
        selectedCustomIds={selectedCustomIds}
        isGuarded={isGuarded}
        hasMetrics={hasMetrics || Boolean(latestMetricsRef.current)}
        supportedClasses={supportedClasses}
        isMetricsModalOpen={isMetricsModalOpen}
        metricsModalTab={metricsModalTab}
        onExecuteActions={handleExecuteCopilotActions}
        onCaptureLiveSnapshot={captureLiveSnapshotFile}
      />
    </div>
  );
}

export default App;
