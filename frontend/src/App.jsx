import React from "react";
import { ControlPanel } from "./components/ControlPanel";
import { DetectionList } from "./components/DetectionList";
import { Header } from "./components/Header";
import { VideoViewport } from "./components/VideoViewport";
import { useWebcamDetection } from "./hooks/useWebcamDetection";
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
    startCamera,
    stopCamera,
    toggleDetection,
    checkBackendHealth,
    handleVideoMetadataLoaded,
  } = useWebcamDetection();

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
          <strong>Backend Offline:</strong> Unable to connect to the FastAPI
          YOLO11 server. Ensure the backend is running on port 8000.
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
            detectionCount={detectionCount}
            inferenceTimeMs={inferenceTimeMs}
            roundTripTimeMs={roundTripTimeMs}
            videoDimensions={videoDimensions}
            onStartCamera={startCamera}
            onStopCamera={stopCamera}
            onToggleDetection={toggleDetection}
          />
        </div>

        <div className="secondary-column">
          <DetectionList
            detections={detections}
            detectionCount={detectionCount}
            inferenceTimeMs={inferenceTimeMs}
            sourceFrameSize={sourceFrameSize}
            cameraStatus={cameraStatus}
            isDetecting={isDetecting}
            supportedClasses={supportedClasses}
            lastUpdatedAt={lastUpdatedAt}
          />
        </div>
      </main>
    </div>
  );
}

export default App;
