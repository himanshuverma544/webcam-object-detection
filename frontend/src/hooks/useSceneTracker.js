import { useCallback, useRef, useState } from "react";
import { playMissingAlertSound, playRestoredAlertSound } from "../utils/audioAlert";

const DEFAULT_MIN_MOVE_DIST = 40; // px
const DEFAULT_SETTLE_DWELL_MS = 1200; // ms
const GRACE_PERIOD_MS = 2500; // ms to confirm missing (filters hand occlusions)
const AUTO_STATIONARY_MIN_MS = 3000; // ms stationary before eligible for auto-missing detection
const MAX_STALE_MISSING_MS = 60000; // Keep missing items in memory up to 60s

function dist(x1, y1, x2, y2) {
  const dx = x2 - x1;
  const dy = y2 - y1;
  return Math.sqrt(dx * dx + dy * dy);
}

export function useSceneTracker() {
  const [minMoveDistance, setMinMoveDistance] = useState(DEFAULT_MIN_MOVE_DIST);
  const [settleDwellTimeMs, setSettleDwellTimeMs] = useState(DEFAULT_SETTLE_DWELL_MS);
  const [isGuarded, setIsGuarded] = useState(false);
  const [guardedCount, setGuardedCount] = useState(0);

  const [activeTracksMap, setActiveTracksMap] = useState({});
  const [missingItems, setMissingItems] = useState([]);
  const [movementEvents, setMovementEvents] = useState([]);
  const [missingEvents, setMissingEvents] = useState([]);

  // Internal mutable state in refs
  const tracksRef = useRef(new Map());
  const trackIdSeqRef = useRef(1);
  const minMoveDistRef = useRef(minMoveDistance);
  const settleDwellRef = useRef(settleDwellTimeMs);
  const isGuardedRef = useRef(isGuarded);

  minMoveDistRef.current = minMoveDistance;
  settleDwellRef.current = settleDwellTimeMs;
  isGuardedRef.current = isGuarded;

  // Toggle Guard Scene Snapshot
  const toggleGuard = useCallback(() => {
    setIsGuarded((prev) => {
      const next = !prev;
      isGuardedRef.current = next;

      const currentTracks = tracksRef.current;
      let count = 0;

      currentTracks.forEach((t) => {
        if (next) {
          if (t.state !== "MISSING") {
            t.isGuarded = true;
            count++;
          }
        } else {
          t.isGuarded = false;
        }
      });

      setGuardedCount(next ? count : 0);
      return next;
    });
  }, []);

  const dismissMissingItem = useCallback((id) => {
    tracksRef.current.delete(id);
    setMissingItems((prev) => prev.filter((item) => item.id !== id));
  }, []);

  const clearEvents = useCallback(() => {
    setMovementEvents([]);
    setMissingEvents([]);
  }, []);

  const resetScene = useCallback(() => {
    tracksRef.current.clear();
    setActiveTracksMap({});
    setMissingItems([]);
    setGuardedCount(0);
    setIsGuarded(false);
  }, []);

  // Main frame processor: called once per frame from detection response
  const processFrame = useCallback((detections) => {
    const now = Date.now();
    const currentDets = Array.isArray(detections) ? detections : [];
    const tracksMap = tracksRef.current;
    const minMove = minMoveDistRef.current;
    const settleTime = settleDwellRef.current;

    const newMoveEvents = [];
    const newMissingEvents = [];
    let audioCue = null;

    // 1. Group current detections by class_id
    const detsByClass = new Map();
    currentDets.forEach((det, idx) => {
      if (!Array.isArray(det.bbox) || det.bbox.length < 4) return;
      const [x1, y1, x2, y2] = det.bbox;
      const cx = Math.round((x1 + x2) / 2);
      const cy = Math.round((y1 + y2) / 2);
      const cId = det.class_id;

      if (!detsByClass.has(cId)) {
        detsByClass.set(cId, []);
      }
      detsByClass.get(cId).push({
        detIdx: idx,
        class_id: cId,
        label: det.label,
        track_id: det.track_id !== undefined && det.track_id !== null ? det.track_id : null,
        bbox: [...det.bbox],
        cx,
        cy,
      });
    });

    // 2. Group existing tracks by class_id
    const tracksByClass = new Map();
    tracksMap.forEach((track, id) => {
      if (!tracksByClass.has(track.class_id)) {
        tracksByClass.set(track.class_id, []);
      }
      tracksByClass.get(track.class_id).push(track);
    });

    // Collect all class IDs present in either detections or tracks
    const allClassIds = new Set([...detsByClass.keys(), ...tracksByClass.keys()]);

    const matchedDetectionIndices = new Set();
    const matchedTrackIds = new Set();

    // 3. For each object class, perform robust matching
    allClassIds.forEach((cId) => {
      const classDets = detsByClass.get(cId) || [];
      const classTracks = tracksByClass.get(cId) || [];

      // A. Priority 1: Match by ByteTrack track_id
      classDets.forEach((d) => {
        if (d.track_id === null || matchedDetectionIndices.has(d.detIdx)) return;
        const matchingTrack = classTracks.find(
          (t) => !matchedTrackIds.has(t.id) && t.track_id === d.track_id
        );
        if (matchingTrack) {
          matchedDetectionIndices.add(d.detIdx);
          matchedTrackIds.add(matchingTrack.id);
          applyDetectionToTrack(matchingTrack, d);
        }
      });

      // B. Priority 2: Greedy nearest-neighbor spatial matching for remaining
      const remainingDets = classDets.filter((d) => !matchedDetectionIndices.has(d.detIdx));
      const remainingTracks = classTracks.filter((t) => !matchedTrackIds.has(t.id));

      if (remainingDets.length > 0 && remainingTracks.length > 0) {
        // Build all candidate pairs with distances
        const pairs = [];
        remainingDets.forEach((d) => {
          remainingTracks.forEach((t) => {
            const dVal = dist(d.cx, d.cy, t.current.x, t.current.y);
            pairs.push({ d, t, dist: dVal });
          });
        });

        // Sort by distance ascending
        pairs.sort((a, b) => a.dist - b.dist);

        pairs.forEach(({ d, t }) => {
          if (matchedDetectionIndices.has(d.detIdx) || matchedTrackIds.has(t.id)) return;
          matchedDetectionIndices.add(d.detIdx);
          matchedTrackIds.add(t.id);
          applyDetectionToTrack(t, d);
        });
      }

      // C. Priority 3: Any unmatched detection is a brand new track
      classDets.forEach((d) => {
        if (matchedDetectionIndices.has(d.detIdx)) return;
        const newId = `trk-${trackIdSeqRef.current++}`;
        const newTrack = {
          id: newId,
          track_id: d.track_id,
          class_id: d.class_id,
          label: d.label,
          origin: { x: d.cx, y: d.cy, bbox: [...d.bbox], timestamp: now },
          current: { x: d.cx, y: d.cy, bbox: [...d.bbox], timestamp: now },
          state: "STATIONARY", // 'STATIONARY' | 'MOVING' | 'MISSING'
          path: [{ x: d.cx, y: d.cy, timestamp: now }],
          settleStart: null,
          lastSeen: now,
          isGuarded: isGuardedRef.current,
          isFlashing: false,
          missingSince: null,
          lastBbox: [...d.bbox],
        };
        tracksMap.set(newId, newTrack);
        matchedTrackIds.add(newId);
      });
    });

    // Helper: update a matched track with its new detection coordinates
    function applyDetectionToTrack(track, d) {
      const prevX = track.current.x;
      const prevY = track.current.y;
      const stepDist = dist(d.cx, d.cy, prevX, prevY);
      const originDist = dist(d.cx, d.cy, track.origin.x, track.origin.y);

      track.lastSeen = now;
      if (d.track_id !== null) {
        track.track_id = d.track_id;
      }
      track.current = { x: d.cx, y: d.cy, bbox: [...d.bbox], timestamp: now };

      // If this track was previously MISSING, it is now RESTORED!
      if (track.state === "MISSING") {
        track.state = "STATIONARY";
        track.origin = { x: d.cx, y: d.cy, bbox: [...d.bbox], timestamp: now };
        track.missingSince = null;
        track.settleStart = null;
        track.path = [{ x: d.cx, y: d.cy, timestamp: now }];

        newMissingEvents.push({
          id: `restored-${track.id}-${now}`,
          type: "RESTORED",
          trackId: track.id,
          label: track.label,
          class_id: track.class_id,
          position: { x: d.cx, y: d.cy },
          timestamp: new Date(now),
          isGuarded: track.isGuarded,
        });
        audioCue = "restored";
        return;
      }

      // If track is currently STATIONARY
      if (track.state === "STATIONARY") {
        if (originDist >= minMove) {
          // Object began moving!
          track.state = "MOVING";
          track.settleStart = null;
          track.path = [
            { x: track.origin.x, y: track.origin.y, timestamp: track.origin.timestamp },
            { x: d.cx, y: d.cy, timestamp: now },
          ];
        }
        return;
      }

      // If track is currently MOVING
      if (track.state === "MOVING") {
        // Append point to trajectory
        if (stepDist > 3 || track.path.length === 0) {
          track.path.push({ x: d.cx, y: d.cy, timestamp: now });
          if (track.path.length > 35) {
            track.path.shift();
          }
        }

        // Check if object is steady / coming to rest
        if (stepDist < 22) {
          if (track.settleStart === null) {
            track.settleStart = now;
          } else if (now - track.settleStart >= settleTime) {
            // Settled at new position! Trigger Relocation Event!
            const totalDist = Math.round(originDist);
            if (totalDist >= minMove) {
              const durationSec = Number(
                Math.max(0.4, (now - (track.path[0]?.timestamp || now)) / 1000).toFixed(1)
              );

              newMoveEvents.push({
                id: `move-${track.id}-${now}`,
                trackId: track.id,
                label: track.label,
                class_id: track.class_id,
                from: { ...track.origin },
                to: { ...track.current },
                distance: totalDist,
                durationSec,
                timestamp: new Date(now),
                trajectory: [...track.path],
              });

              track.isFlashing = true;
              setTimeout(() => {
                track.isFlashing = false;
              }, 2200);
            }

            // Reset origin to current settled spot
            track.origin = { x: d.cx, y: d.cy, bbox: [...d.bbox], timestamp: now };
            track.state = "STATIONARY";
            track.settleStart = null;
            track.path = [{ x: d.cx, y: d.cy, timestamp: now }];
          }
        } else {
          // In active motion
          track.settleStart = null;
        }
      }
    }

    // 4. Handle unmatched tracks (items that were NOT detected this frame)
    tracksMap.forEach((track, id) => {
      if (matchedTrackIds.has(id)) return;

      const unseenMs = now - track.lastSeen;

      // If item was in active transit/moving, give it 2 seconds of grace (hand occluding item)
      if (track.state === "MOVING" && unseenMs < 2000) {
        return;
      }

      // Check if item should become MISSING
      if (track.state !== "MISSING") {
        const isEligible =
          track.isGuarded ||
          (track.origin && now - track.origin.timestamp >= AUTO_STATIONARY_MIN_MS);

        if (isEligible && unseenMs >= GRACE_PERIOD_MS) {
          // CONFIRMED MISSING!
          track.state = "MISSING";
          track.missingSince = new Date(track.lastSeen);
          track.lastBbox = [...track.current.bbox];

          newMissingEvents.push({
            id: `missing-${track.id}-${now}`,
            type: "MISSING",
            trackId: track.id,
            label: track.label,
            class_id: track.class_id,
            position: { x: track.current.x, y: track.current.y },
            timestamp: new Date(now),
            isGuarded: track.isGuarded,
          });
          audioCue = "missing";
        }
      } else {
        // Already missing: prune if absent for too long and not guarded
        if (!track.isGuarded && unseenMs > MAX_STALE_MISSING_MS) {
          tracksMap.delete(id);
        }
      }
    });

    // 5. Play audio cues
    if (audioCue === "missing") {
      playMissingAlertSound();
    } else if (audioCue === "restored") {
      playRestoredAlertSound();
    }

    // 6. Generate snapshots for React state
    const activeMap = {};
    const activeMissingList = [];

    tracksMap.forEach((t) => {
      if (t.state === "MISSING") {
        activeMissingList.push({
          id: t.id,
          label: t.label,
          class_id: t.class_id,
          lastBbox: t.lastBbox,
          centroid: { x: t.current.x, y: t.current.y },
          missingSince: t.missingSince,
          lastSeen: new Date(t.lastSeen),
          isGuarded: t.isGuarded,
        });
      } else {
        activeMap[t.id] = {
          id: t.id,
          track_id: t.track_id,
          label: t.label,
          class_id: t.class_id,
          state: t.state,
          origin: t.origin,
          current: t.current,
          path: t.path,
          distance: Math.round(dist(t.current.x, t.current.y, t.origin.x, t.origin.y)),
          isFlashing: t.isFlashing,
          isGuarded: t.isGuarded,
        };
      }
    });

    setActiveTracksMap(activeMap);
    setMissingItems(activeMissingList);

    if (newMoveEvents.length > 0) {
      setMovementEvents((prev) => [...newMoveEvents, ...prev].slice(0, 50));
    }
    if (newMissingEvents.length > 0) {
      setMissingEvents((prev) => [...newMissingEvents, ...prev].slice(0, 50));
    }
  }, []);

  const activeMovingCount = Object.values(activeTracksMap).filter(
    (t) => t.state === "MOVING"
  ).length;

  return {
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
  };
}

