import React, { useCallback, useEffect, useState } from "react";
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

  // Session metrics accumulator
  const {
    completedMetrics,
    hasMetrics,
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
    snapshotSession();
    stopCamera();
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
    (actions) => {
      if (!Array.isArray(actions)) return;

      for (const action of actions) {
        if (!action || !action.type) continue;
        const payload = action.payload || {};

        switch (action.type) {
          case "START_CAMERA":
            startCamera();
            break;

          case "STOP_CAMERA":
            handleStopCamera();
            break;

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
            const itemRef = String(payload.item_ref || "").toLowerCase();
            const target = customItems.find(
              (it) => it.id === itemRef || it.name.toLowerCase() === itemRef
            );
            if (target) {
              removeCustomItem(target.id);
            }
            break;
          }

          case "SET_SIMILARITY_THRESHOLD":
            if (typeof payload.value === "number") {
              setSimilarityThreshold(payload.value);
            }
            break;

          case "CAPTURE_CUSTOM_ITEM": {
            const itemName = payload.name || "Custom Item";
            if (cameraStatus === "active") {
              captureLiveSnapshotFile(itemName).then((file) => {
                if (file) {
                  uploadCustomItem(itemName, [file], null);
                }
              });
            } else {
              startCamera();
              setTimeout(() => {
                captureLiveSnapshotFile(itemName).then((file) => {
                  if (file) {
                    uploadCustomItem(itemName, [file], null);
                  }
                });
              }, 1200);
            }
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

          case "SHOW_METRICS":
            setIsMetricsModalOpen(true);
            break;

          case "SEND_METRICS_EMAIL":
            if (completedMetrics) {
              sendMetricsEmail(completedMetrics, payload.recipient_email);
            }
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
            hasMetrics={hasMetrics}
            onShowMetrics={() => setIsMetricsModalOpen(true)}
            onStartCamera={startCamera}
            onStopCamera={handleStopCamera}
            onToggleDetection={toggleDetection}
          />
        </div>

        <div className="secondary-column">
          <DetectionList
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
        metrics={completedMetrics}
      />

      <ChatCopilot
        cameraStatus={cameraStatus}
        isDetecting={isDetecting}
        confidence={confidence}
        selectedClasses={selectedClasses}
        customItems={customItems}
        selectedCustomIds={selectedCustomIds}
        isGuarded={isGuarded}
        supportedClasses={supportedClasses}
        onExecuteActions={handleExecuteCopilotActions}
        onCaptureLiveSnapshot={captureLiveSnapshotFile}
      />
    </div>
  );
}

export default App;
