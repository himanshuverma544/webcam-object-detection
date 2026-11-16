import { useCallback, useEffect, useRef, useState } from "react";

function formatDuration(ms) {
  if (!ms || ms < 1000) return "< 1s";
  const totalSec = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSec / 60);
  const seconds = totalSec % 60;
  if (minutes === 0) return `${seconds}s`;
  return `${minutes}m ${seconds}s`;
}

export function useSessionMetrics({
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
}) {
  const [completedMetrics, setCompletedMetrics] = useState(null);

  // Active session tracking state in refs (to avoid excessive re-renders during 30 FPS loops)
  const sessionActiveRef = useRef(false);
  const sessionStartRef = useRef(null);
  const frameCountRef = useRef(0);
  const totalInferenceMsRef = useRef(0);
  const inferenceSamplesRef = useRef(0);
  const totalRttMsRef = useRef(0);
  const rttSamplesRef = useRef(0);
  const peakObjectsRef = useRef(0);
  const classCountsRef = useRef({});
  const uniqueTrackIdsRef = useRef(new Set());

  // Guard duration tracking
  const guardStartRef = useRef(null);
  const totalGuardedMsRef = useRef(0);

  // Guard tracking effect
  useEffect(() => {
    if (cameraStatus === "active" && isGuarded) {
      guardStartRef.current = Date.now();
    } else if (guardStartRef.current) {
      totalGuardedMsRef.current += Date.now() - guardStartRef.current;
      guardStartRef.current = null;
    }
  }, [cameraStatus, isGuarded]);

  // When camera turns active: initialize a fresh session tracker
  useEffect(() => {
    if (cameraStatus === "active") {
      sessionActiveRef.current = true;
      sessionStartRef.current = Date.now();
      frameCountRef.current = 0;
      totalInferenceMsRef.current = 0;
      inferenceSamplesRef.current = 0;
      totalRttMsRef.current = 0;
      rttSamplesRef.current = 0;
      peakObjectsRef.current = 0;
      classCountsRef.current = {};
      uniqueTrackIdsRef.current = new Set();
      guardStartRef.current = isGuarded ? Date.now() : null;
      totalGuardedMsRef.current = 0;
    }
  }, [cameraStatus]); // eslint-disable-line react-hooks/exhaustive-deps

  // Frame update aggregator
  useEffect(() => {
    if (!sessionActiveRef.current || cameraStatus !== "active") return;

    frameCountRef.current += 1;

    if (typeof inferenceTimeMs === "number" && inferenceTimeMs > 0) {
      totalInferenceMsRef.current += inferenceTimeMs;
      inferenceSamplesRef.current += 1;
    }

    if (typeof roundTripTimeMs === "number" && roundTripTimeMs > 0) {
      totalRttMsRef.current += roundTripTimeMs;
      rttSamplesRef.current += 1;
    }

    const currentCount = Array.isArray(detections) ? detections.length : detectionCount || 0;
    if (currentCount > peakObjectsRef.current) {
      peakObjectsRef.current = currentCount;
    }

    if (Array.isArray(detections)) {
      detections.forEach((det) => {
        if (det.label) {
          classCountsRef.current[det.label] = (classCountsRef.current[det.label] || 0) + 1;
        }
        if (det.track_id !== undefined && det.track_id !== null) {
          uniqueTrackIdsRef.current.add(det.track_id);
        }
      });
    }
  }, [lastUpdatedAt]); // eslint-disable-line react-hooks/exhaustive-deps

  // Compute snapshot of metrics from current session
  const snapshotSession = useCallback(() => {
    if (!sessionStartRef.current) return null;

    const sessionEndTime = Date.now();
    const durationMs = Math.max(1000, sessionEndTime - sessionStartRef.current);

    // Finalize guard time
    let guardedMs = totalGuardedMsRef.current;
    if (guardStartRef.current) {
      guardedMs += sessionEndTime - guardStartRef.current;
    }

    const frameCount = frameCountRef.current;
    const avgInference =
      inferenceSamplesRef.current > 0
        ? (totalInferenceMsRef.current / inferenceSamplesRef.current).toFixed(1)
        : null;

    const avgRtt =
      rttSamplesRef.current > 0
        ? (totalRttMsRef.current / rttSamplesRef.current).toFixed(1)
        : null;

    const avgFps =
      durationMs > 0
        ? (frameCount / (durationMs / 1000)).toFixed(1)
        : "0.0";

    // Top detected classes
    const classEntries = Object.entries(classCountsRef.current);
    const totalDetections = classEntries.reduce((acc, [, count]) => acc + count, 0);
    const topClasses = classEntries
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([label, count]) => ({
        label,
        count,
        percent: totalDetections > 0 ? Math.round((count / totalDetections) * 100) : 0,
      }));

    // Unique entities
    let uniqueEntitiesCount = uniqueTrackIdsRef.current.size;
    if (activeTracksMap && typeof activeTracksMap === "object") {
      uniqueEntitiesCount = Math.max(
        uniqueEntitiesCount,
        Object.keys(activeTracksMap).length
      );
    }

    // Unresolved missing items snapshot
    const currentMissing = Array.isArray(missingItems) ? [...missingItems] : [];
    const missingEventsList = Array.isArray(missingEvents) ? [...missingEvents] : [];
    const movementEventsList = Array.isArray(movementEvents) ? [...movementEvents] : [];

    const restoredCount = missingEventsList.filter((e) => e.type === "RESTORED").length;
    const missingIncidentsCount = missingEventsList.filter((e) => e.type === "MISSING").length;

    // Most active moved object
    const movesByTrack = {};
    movementEventsList.forEach((e) => {
      const key = `${e.label || "Object"} (#${e.trackId})`;
      if (!movesByTrack[key]) {
        movesByTrack[key] = { label: e.label || "Object", trackId: e.trackId, moves: 0, distance: 0 };
      }
      movesByTrack[key].moves += 1;
      movesByTrack[key].distance += e.distance || 0;
    });

    const activeList = Object.values(movesByTrack).sort((a, b) => b.moves - a.moves);
    const mostActiveObject = activeList.length > 0 ? activeList[0] : null;

    // Stability calculation
    const stabilityRatio =
      movementEventsList.length === 0
        ? "100% Still (No Displacements)"
        : `${Math.max(60, Math.round(100 - (movementEventsList.length / Math.max(1, frameCount)) * 100))}% Still`;

    // Security Status Tag
    let securityStatus = "CLEAN";
    let statusMessage = "No Security Incidents";
    if (currentMissing.length > 0) {
      securityStatus = "ALERT";
      statusMessage = `${currentMissing.length} Unresolved Missing Item${currentMissing.length > 1 ? "s" : ""}`;
    } else if (missingIncidentsCount > 0 || movementEventsList.length > 0) {
      securityStatus = "WARNING";
      statusMessage = `${movementEventsList.length + missingIncidentsCount} Activity Alert${
        movementEventsList.length + missingIncidentsCount > 1 ? "s" : ""
      } Recorded`;
    }

    const guardPercent = Math.min(100, Math.round((guardedMs / durationMs) * 100));

    const metricsSnapshot = {
      timestamp: new Date().toLocaleString(),
      durationMs,
      durationFormatted: formatDuration(durationMs),
      guardedMs,
      guardedFormatted:
        guardedMs > 500
          ? `${formatDuration(guardedMs)} (${guardPercent}% of session)`
          : "Not Armed",
      isGuardedAtEnd: Boolean(isGuarded),
      frameCount,
      avgFps: `${avgFps} FPS`,
      avgInferenceMs: avgInference ? `${avgInference} ms` : "—",
      avgRttMs: avgRtt ? `${avgRtt} ms` : "—",
      confidenceThreshold: `${Math.round(confidence * 100)}%`,
      peakObjectCount: peakObjectsRef.current,
      uniqueEntitiesCount,
      topClasses,
      // Security
      securityStatus,
      statusMessage,
      missingIncidentsCount,
      restoredCount,
      unresolvedMissing: currentMissing,
      unresolvedMissingCount: currentMissing.length,
      movementEventsCount: movementEventsList.length,
      mostActiveObject,
      stabilityRatio,
    };

    setCompletedMetrics(metricsSnapshot);
    sessionActiveRef.current = false;
    return metricsSnapshot;
  }, [
    isGuarded,
    confidence,
    missingItems,
    missingEvents,
    movementEvents,
    activeTracksMap,
  ]);

  return {
    completedMetrics,
    hasMetrics: Boolean(completedMetrics),
    snapshotSession,
    clearMetrics: () => setCompletedMetrics(null),
  };
}

