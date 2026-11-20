/**
 * Generates an executive-level plain text / markdown session summary
 * for the Webcam Object Detection system.
 * Formats all dates as 'DD MMM YYYY' (e.g. 27 May 2027) and
 * all times in 24-hour UTC ('HH:mm:ss UTC').
 */

const MONTHS = [
  "Jan",
  "Feb",
  "Mar",
  "Apr",
  "May",
  "Jun",
  "Jul",
  "Aug",
  "Sep",
  "Oct",
  "Nov",
  "Dec",
];

export function formatDateUTC(val) {
  if (!val) return "Unknown Date";
  const d = val instanceof Date ? val : new Date(val);
  if (isNaN(d.getTime())) return "Unknown Date";
  const day = d.getUTCDate();
  const month = MONTHS[d.getUTCMonth()];
  const year = d.getUTCFullYear();
  return `${day} ${month} ${year}`;
}

export function formatTimeUTC(val) {
  if (!val) return "—";
  const d = val instanceof Date ? val : new Date(val);
  if (isNaN(d.getTime())) return "—";
  const hours = String(d.getUTCHours()).padStart(2, "0");
  const mins = String(d.getUTCMinutes()).padStart(2, "0");
  const secs = String(d.getUTCSeconds()).padStart(2, "0");
  return `${hours}:${mins}:${secs} UTC`;
}

export function formatDateTimeUTC(val) {
  if (!val) return "—";
  const d = val instanceof Date ? val : new Date(val);
  if (isNaN(d.getTime())) return "—";
  return `${formatDateUTC(d)} ${formatTimeUTC(d)}`;
}

export function formatDurationMs(ms) {
  if (typeof ms !== "number" || isNaN(ms) || ms < 1000) return "< 1s";
  const totalSec = Math.floor(ms / 1000);
  const hours = Math.floor(totalSec / 3600);
  const minutes = Math.floor((totalSec % 3600) / 60);
  const seconds = totalSec % 60;
  if (hours > 0) return `${hours}h ${minutes}m ${seconds}s`;
  if (minutes === 0) return `${seconds}s`;
  return `${minutes}m ${seconds}s`;
}

export function generateExecutiveSummary(metrics) {
  if (!metrics) {
    return "No session metrics recorded.";
  }

  const now = Date.now();
  const sessionEnd = metrics.sessionEndMs || now;
  const sessionStart = metrics.sessionStartMs || sessionEnd - (metrics.durationMs || 0);

  const durationFormatted = metrics.durationFormatted || formatDurationMs(metrics.durationMs);
  const frameCount = metrics.frameCount || 0;
  const avgFps = metrics.avgFps || "0.0 FPS";

  // Unresolved missing items
  const unresolved = Array.isArray(metrics.unresolvedMissing)
    ? metrics.unresolvedMissing
    : [];
  const unresolvedCount = unresolved.length;

  // Missing & Restored events log
  const missingEvents = Array.isArray(metrics.missingEvents)
    ? metrics.missingEvents
    : [];

  const restoredEvents = missingEvents.filter((e) => e.type === "RESTORED");
  const restoredCount = restoredEvents.length;
  const missingCount = metrics.missingIncidentsCount || missingEvents.filter((e) => e.type === "MISSING").length;

  // Movement events log
  const movementEvents = Array.isArray(metrics.movementEvents)
    ? metrics.movementEvents
    : [];
  const movementEventsCount = metrics.movementEventsCount || movementEvents.length;

  // Group movement by item
  const movesByItem = {};
  movementEvents.forEach((evt) => {
    const trackId = evt.trackId !== undefined && evt.trackId !== null ? evt.trackId : "—";
    const label = evt.label || "Object";
    const key = `${label} (#${trackId})`;

    if (!movesByItem[key]) {
      movesByItem[key] = {
        label,
        trackId,
        count: 0,
        totalDistance: 0,
        events: [],
      };
    }
    movesByItem[key].count += 1;
    movesByItem[key].totalDistance += Math.round(evt.distance || 0);
    movesByItem[key].events.push(evt);
  });

  const movedItemsList = Object.values(movesByItem).sort((a, b) => b.count - a.count);
  const movedItemNames = movedItemsList.map((m) => m.label);
  const movedItemsSet = new Set(movedItemNames.map((n) => n.toLowerCase()));

  // Active / Top classes
  const topClasses = Array.isArray(metrics.topClasses) ? metrics.topClasses : [];
  const activeClasses = Array.isArray(metrics.selectedClasses) && metrics.selectedClasses.length > 0
    ? metrics.selectedClasses.join(", ")
    : topClasses.map((c) => c.label).join(", ") || "None";

  // Static / Unmoved items (among top detected)
  const staticClasses = topClasses
    .filter((c) => !movedItemsSet.has(c.label.toLowerCase()))
    .map((c) => c.label);

  // 1. Executive Summary Narrative
  let executiveSummary = "";
  if (frameCount === 0) {
    executiveSummary = `Completed a brief camera session (${durationFormatted}). No detection frames were processed (session was stopped quickly or 0 COCO classes were selected).`;
  } else if (unresolvedCount > 0) {
    const unresolvedNames = unresolved.map((u) => u.label).join(", ");
    executiveSummary = `Surveillance session ran for ${durationFormatted} across ${frameCount} frames (${avgFps}) under ${metrics.guardedFormatted !== "Not Armed" ? "active" : "unarmed"} Guard Mode.\nSECURITY ALERT: ${unresolvedCount} guarded asset${unresolvedCount > 1 ? "s" : ""} (${unresolvedNames}) ${unresolvedCount > 1 ? "remain" : "remains"} unresolved missing. ${restoredCount > 0 ? `${restoredCount} asset(s) were temporarily absent and recovered. ` : ""}${movedItemsList.length > 0 ? `Physical movement was registered across ${movedItemsList.length} item(s) with ${movementEventsCount} total relocation events.` : "No other items moved."}`;
  } else if (restoredCount > 0) {
    executiveSummary = `Surveillance session ran for ${durationFormatted} across ${frameCount} frames (${avgFps}).\nPerimeter alert triggered: ${restoredCount} item(s) were temporarily absent but all were verified restored. ${movedItemsList.length > 0 ? `Physical movement was registered across ${movedItemsList.length} item(s) with ${movementEventsCount} relocation events.` : "Scene is now stable."} Zero unresolved missing assets remain.`;
  } else if (movementEventsCount > 0) {
    executiveSummary = `Active surveillance session ran for ${durationFormatted} across ${frameCount} frames (${avgFps}).\nRegistered ${movementEventsCount} physical relocation events across ${movedItemsList.length} item(s). Perimeter remained secure with 0 missing assets and ${metrics.stabilityRatio || "100% Still"} scene stability.`;
  } else {
    executiveSummary = `Surveillance session completed for ${durationFormatted} across ${frameCount} frames (${avgFps}) with ${metrics.guardedFormatted !== "Not Armed" ? "Guard Mode armed" : "Guard Mode unarmed"}.\nScene remained 100% still and secure with zero movement infractions and zero missing items.`;
  }

  // 2. Missing Items Section
  let securityLines = [];
  const statusStr = unresolvedCount > 0 ? `ALERT (${unresolvedCount} Unresolved Missing Asset${unresolvedCount > 1 ? "s" : ""})` : metrics.securityStatus || "CLEAN";
  securityLines.push(`* **Active Status**: ${statusStr}`);

  if (unresolvedCount > 0) {
    securityLines.push(`* **Unresolved Missing Items**:`);
    unresolved.forEach((item) => {
      const missingStart = item.missingSince ? new Date(item.missingSince).getTime() : sessionEnd;
      const missingDurMs = Math.max(0, sessionEnd - missingStart);
      const trackStr = item.id ? ` (Track #${String(item.id).replace(/^track-/, "")})` : "";
      securityLines.push(
        `  * **${item.label}**${trackStr}: Missing for **${formatDurationMs(missingDurMs)}** (Disappeared at ${formatTimeUTC(item.missingSince || sessionStart)}) — *NOT RECOVERED*`
      );
    });
  } else {
    securityLines.push(`* **Unresolved Missing Items**: 0 (None)`);
  }

  if (restoredCount > 0) {
    securityLines.push(`* **Restored / Recovered Items**:`);
    restoredEvents.forEach((rEvt, idx) => {
      // Find matching earlier MISSING event for this track
      const priorMissing = missingEvents
        .slice(0, missingEvents.indexOf(rEvt))
        .reverse()
        .find((e) => e.trackId === rEvt.trackId && e.type === "MISSING");

      const trackStr = rEvt.trackId ? ` (Track #${String(rEvt.trackId).replace(/^track-/, "")})` : "";
      if (priorMissing && priorMissing.timestamp && rEvt.timestamp) {
        const awayDur = Math.max(0, new Date(rEvt.timestamp).getTime() - new Date(priorMissing.timestamp).getTime());
        securityLines.push(
          `  * **${rEvt.label}**${trackStr}: Was missing for **${formatDurationMs(awayDur)}** (Disappeared at ${formatTimeUTC(priorMissing.timestamp)}, Restored at ${formatTimeUTC(rEvt.timestamp)})`
        );
      } else {
        securityLines.push(
          `  * **${rEvt.label}**${trackStr}: Restored at ${formatTimeUTC(rEvt.timestamp)}`
        );
      }
    });
  } else {
    securityLines.push(`* **Restored Items**: 0`);
  }

  securityLines.push(`* **Total Disappearance Incidents**: ${missingCount} event${missingCount === 1 ? "" : "s"} recorded`);
  securityLines.push(`* **Guard Coverage**: ${metrics.guardedFormatted || "Not Armed"}`);

  // 3. Movement Breakdown Section
  let movementLines = [];
  movementLines.push(`* **Total Relocations**: ${movementEventsCount} movement event${movementEventsCount === 1 ? "" : "s"} registered`);
  movementLines.push(`* **Scene Stability**: ${metrics.stabilityRatio || "100% Still"}`);

  if (movedItemsList.length > 0) {
    movementLines.push(`* **Moved Items Detail**:`);
    movedItemsList.forEach((m) => {
      const trackStr = m.trackId !== "—" ? ` (Track #${String(m.trackId).replace(/^track-/, "")})` : "";
      movementLines.push(
        `  * **${m.label}**${trackStr}:\n    * Moved **${m.count} time${m.count > 1 ? "s" : ""}** | Total Displacement: **${m.totalDistance} px**\n    * Relocation Log (UTC):\n${m.events.map((e) => `      - ${formatTimeUTC(e.timestamp)} (${Math.round(e.distance || 0)} px displacement)`).join("\n")}`
      );
    });
  } else {
    movementLines.push(`* **Moved Items Detail**: None`);
  }

  movementLines.push(`* **Static Items (0 Movement)**: ${staticClasses.length > 0 ? staticClasses.join(", ") : "None"}`);

  // 4. Takeaway
  let takeaway = "";
  if (frameCount === 0) {
    takeaway = "Session ended without active frames. Select one or more COCO class pills and allow the camera to run with objects in view to collect live metrics.";
  } else if (unresolvedCount > 0) {
    const unNames = unresolved.map((u) => u.label).join(", ");
    takeaway = `CRITICAL: **${unNames}** breached the guarded perimeter and has not returned. Review video logs and timestamps corresponding to the disappearance events.`;
  } else if (restoredCount > 0) {
    takeaway = `All previously displaced or occluded items have returned to scene view. Security integrity verified with 0 unresolved breaches.`;
  } else if (movementEventsCount > 0) {
    const mostActive = metrics.mostActiveObject ? `${metrics.mostActiveObject.label} (${metrics.mostActiveObject.moves} moves)` : "detected items";
    takeaway = `Physical movement was detected (most active: ${mostActive}), but all items settled safely within camera view. Scene stability verified at ${metrics.stabilityRatio || "nominal"}.`;
  } else {
    takeaway = "Scene remained fully secure throughout the surveillance period. All tracked inventory is accounted for with nominal inference latency and zero perimeter infractions.";
  }

  // Final Structured Markdown Output
  return `
# WEBCAM OBJECT DETECTION SESSION REPORT
Generated: ${formatDateUTC(sessionEnd)}, ${formatTimeUTC(sessionEnd)}

### Executive Summary
${executiveSummary}

---

### Security & Missing Items Breakdown
${securityLines.join("\n")}

---

### Object Movement & Relocation Breakdown
${movementLines.join("\n")}

---

### Session Profile & System Performance
* **Session Duration**: ${durationFormatted} (Start: ${formatTimeUTC(sessionStart)} | End: ${formatTimeUTC(sessionEnd)})
* **Frames Processed**: ${frameCount} frames (Effective Cadence: ${avgFps})
* **Monitored COCO Classes**: ${activeClasses}
* **Peak Objects in View**: ${metrics.peakObjectCount || 0} simultaneous
* **Avg YOLO Inference**: ${metrics.avgInferenceMs || "—"}
* **Avg Round-Trip Latency**: ${metrics.avgRttMs || "—"}
* **Confidence Threshold**: ${metrics.confidenceThreshold || "—"}

---

### Takeaway & Action Items
${takeaway}
`.trim();
}
