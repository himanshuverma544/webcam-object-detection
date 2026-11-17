import React, { useCallback, useEffect, useState } from "react";
import { ControlPanel } from "./components/ControlPanel";
import { DetectionList } from "./components/DetectionList";
import { Header } from "./components/Header";
import { MetricsModal } from "./components/MetricsModal";
import { MissingAlertBanner } from "./components/MissingAlertBanner";
import { VideoViewport } from "./components/VideoViewport";
import { useSceneTracker } from "./hooks/useSceneTracker";
import { useSessionMetrics } from "./hooks/useSessionMetrics";
import { useWebcamDetection } from "./hooks/useWebcamDetection";
import { API_BASE_URL } from "./services/api";
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
    startCamera,
    stopCamera,
    toggleDetection,
    checkBackendHealth,
    handleVideoMetadataLoaded,
  } = useWebcamDetection();

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
          />
        </div>
      </main>

      <MetricsModal
        isOpen={isMetricsModalOpen}
        onClose={() => setIsMetricsModalOpen(false)}
        metrics={completedMetrics}
      />
    </div>
  );
}

export default App;
