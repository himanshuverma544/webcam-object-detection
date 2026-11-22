import React, { useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import { sendMetricsEmail } from "../services/api";
import { generateExecutiveSummary } from "../utils/summaryGenerator";

const DEFAULT_RECIPIENT = (import.meta.env.VITE_DEFAULT_RECIPIENT_EMAIL || "").trim();

export function MetricsModal({ isOpen, onClose, metrics, activeTab = "metrics", isCopiedExternal = 0 }) {
  const [copied, setCopied] = useState(false);
  const [viewMode, setViewMode] = useState(activeTab || "metrics"); // 'metrics' | 'summary'
  const [emailStatus, setEmailStatus] = useState("idle"); // 'idle' | 'sending' | 'success' | 'error'
  const [emailMessage, setEmailMessage] = useState("");

  // Close on Escape key
  useEffect(() => {
    if (!isOpen) return;
    const handleKeyDown = (e) => {
      if (e.key === "Escape") {
        onClose();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose]);

  // Sync active tab and reset status when opened or activeTab switches
  useEffect(() => {
    if (isOpen) {
      setEmailStatus("idle");
      setEmailMessage("");
      setCopied(false);
      setViewMode(activeTab || "metrics");
    }
  }, [isOpen, activeTab]);

  // Flash copied state when triggered externally (e.g. via AI Copilot)
  useEffect(() => {
    if (isCopiedExternal) {
      setCopied(true);
      const timer = setTimeout(() => setCopied(false), 2000);
      return () => clearTimeout(timer);
    }
  }, [isCopiedExternal]);

  const summaryText = useMemo(() => {
    return metrics ? generateExecutiveSummary(metrics) : "";
  }, [metrics]);

  if (!isOpen || !metrics) {
    return null;
  }

  const handleCopy = () => {
    navigator.clipboard.writeText(summaryText).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }).catch(() => {});
  };

  const handleSendEmail = async () => {
    setEmailStatus("sending");
    setEmailMessage("");
    try {
      const res = await sendMetricsEmail(metrics, DEFAULT_RECIPIENT || undefined);
      setEmailStatus("success");
      setEmailMessage(res.message || "Metrics report successfully dispatched");
    } catch (err) {
      setEmailStatus("error");
      setEmailMessage(err.message || "Failed to send email. Check backend settings.");
    }
  };

  const getStatusBadgeClass = () => {
    if (metrics.securityStatus === "ALERT") return "badge-alert";
    if (metrics.securityStatus === "WARNING") return "badge-warning";
    return "badge-clean";
  };

  const getStatusPrefix = () => {
    if (metrics.securityStatus === "ALERT") return "[ALERT] ";
    if (metrics.securityStatus === "WARNING") return "[WARNING] ";
    return "[CLEAN] ";
  };

  return (
    <div className="modal-backdrop" onClick={onClose} role="dialog" aria-modal="true">
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        {/* Modal Header */}
        <div className="modal-header">
          <div className="modal-header-info">
            <div className="modal-title-row">
              <h2>
                {viewMode === "summary"
                  ? "Executive Session Summary"
                  : "Session Summary & Detection Metrics"}
              </h2>
            </div>
            <p className="modal-subtitle">
              {viewMode === "summary" ? (
                <>
                  Format: <strong>24-Hour UTC</strong> &bull; Total Session: <strong>{metrics.durationFormatted}</strong>
                </>
              ) : (
                <>
                  Recorded: <strong>{metrics.timestamp}</strong> &bull; Total Session: <strong>{metrics.durationFormatted}</strong>
                </>
              )}
            </p>
          </div>
          <div className="modal-header-actions">
            <span className={`metrics-badge ${getStatusBadgeClass()}`}>
              {getStatusPrefix()}
              {metrics.statusMessage}
            </span>
            <button
              type="button"
              className="modal-close-btn"
              onClick={onClose}
              aria-label="Close modal"
            >
              &times;
            </button>
          </div>
        </div>

        {viewMode === "summary" ? (
          <div className="modal-summary-body">
            <div className="summary-paper">
              <div className="summary-paper-header">
                <span className="summary-paper-badge">Executive Session Report</span>
                <span className="summary-paper-date">24-Hour UTC Format</span>
              </div>
              <div className="summary-rendered-markdown">
                <ReactMarkdown>{summaryText}</ReactMarkdown>
              </div>
            </div>
          </div>
        ) : (
          <>
            {/* Quick KPI Overview Tiles */}
            <div className="modal-kpi-grid">
          <div className="modal-kpi-card">
            <span className="kpi-label">Session Duration</span>
            <span className="kpi-value">{metrics.durationFormatted}</span>
            <span className="kpi-subtext">Guarded: {metrics.guardedFormatted}</span>
          </div>

          <div className="modal-kpi-card">
            <span className="kpi-label">Security Alerts</span>
            <span className={`kpi-value ${metrics.unresolvedMissingCount > 0 ? "text-danger" : ""}`}>
              {metrics.missingIncidentsCount + metrics.movementEventsCount}
            </span>
            <span className="kpi-subtext">
              {metrics.unresolvedMissingCount > 0
                ? `${metrics.unresolvedMissingCount} item(s) missing`
                : "All items accounted for"}
            </span>
          </div>

          <div className="modal-kpi-card">
            <span className="kpi-label">Unique Entities</span>
            <span className="kpi-value">{metrics.uniqueEntitiesCount}</span>
            <span className="kpi-subtext">Peak in frame: {metrics.peakObjectCount}</span>
          </div>

          <div className="modal-kpi-card">
            <span className="kpi-label">Avg Backend Latency</span>
            <span className="kpi-value text-accent">{metrics.avgInferenceMs}</span>
            <span className="kpi-subtext">Rate: {metrics.avgFps}</span>
          </div>
        </div>

        {/* Tabular Metrics Content */}
        <div className="modal-body-scroll">
          {/* Table 1: Security & Incident Alerts */}
          <section className="metrics-section">
            <div className="section-title-row">
              <h3>Security &amp; Incident Alerts</h3>
              <span className="section-tag">Surveillance &amp; Guard</span>
            </div>
            <div className="table-wrapper">
              <table className="metrics-table">
                <thead>
                  <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Status &amp; Details</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td><strong>Missing Item Incidents</strong></td>
                    <td className="metric-val">{metrics.missingIncidentsCount}</td>
                    <td>
                      {metrics.missingIncidentsCount > 0
                        ? `${metrics.missingIncidentsCount} item disappearance events registered`
                        : "No missing item alerts triggered"}
                    </td>
                  </tr>
                  <tr className={metrics.unresolvedMissingCount > 0 ? "row-danger" : ""}>
                    <td><strong>Unresolved Missing Items</strong></td>
                    <td className="metric-val">
                      <span className={metrics.unresolvedMissingCount > 0 ? "badge-danger" : ""}>
                        {metrics.unresolvedMissingCount}
                      </span>
                    </td>
                    <td>
                      {metrics.unresolvedMissingCount > 0 ? (
                        <span className="text-danger font-semibold">
                          Still missing: {metrics.unresolvedMissing.map((i) => i.label).join(", ")}
                        </span>
                      ) : (
                        "None — all guarded items in place"
                      )}
                    </td>
                  </tr>
                  <tr>
                    <td><strong>Restored Items</strong></td>
                    <td className="metric-val">{metrics.restoredCount}</td>
                    <td>
                      {metrics.restoredCount > 0
                        ? `${metrics.restoredCount} previously missing item(s) returned to view`
                        : "No restoration events"}
                    </td>
                  </tr>
                  <tr>
                    <td><strong>Movement &amp; Tamper Alerts</strong></td>
                    <td className="metric-val">{metrics.movementEventsCount}</td>
                    <td>
                      {metrics.movementEventsCount > 0
                        ? `${metrics.movementEventsCount} displacement events detected`
                        : "No movement threshold violations"}
                    </td>
                  </tr>
                  <tr>
                    <td><strong>Scene Guard Protection</strong></td>
                    <td className="metric-val">{metrics.guardedFormatted}</td>
                    <td>
                      {metrics.isGuardedAtEnd
                        ? "Guard Mode was actively armed at session end"
                        : "Guard Mode was disarmed or inactive"}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </section>

          {/* Table 2: Activity & Object Analytics */}
          <section className="metrics-section">
            <div className="section-title-row">
              <h3>Activity &amp; Object Analytics</h3>
              <span className="section-tag">Detection Insights</span>
            </div>
            <div className="table-wrapper">
              <table className="metrics-table">
                <thead>
                  <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Details &amp; Breakdown</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td><strong>Session Active Duration</strong></td>
                    <td className="metric-val">{metrics.durationFormatted}</td>
                    <td>Continuous webcam monitoring time</td>
                  </tr>
                  <tr>
                    <td><strong>Peak Simultaneous Objects</strong></td>
                    <td className="metric-val">{metrics.peakObjectCount}</td>
                    <td>Highest number of objects detected concurrently in any single frame</td>
                  </tr>
                  <tr>
                    <td><strong>Total Tracked Entities</strong></td>
                    <td className="metric-val">{metrics.uniqueEntitiesCount}</td>
                    <td>Distinct object IDs identified across the session</td>
                  </tr>
                  <tr>
                    <td><strong>Top Detected Classes</strong></td>
                    <td className="metric-val" colSpan={2}>
                      {metrics.topClasses && metrics.topClasses.length > 0 ? (
                        <div className="class-pills-list">
                          {metrics.topClasses.map((c) => (
                            <span key={c.label} className="class-pill">
                              <strong>{c.label}</strong> ({c.percent}% &bull; {c.count} frames)
                            </span>
                          ))}
                        </div>
                      ) : (
                        <span className="text-muted">No objects detected during session</span>
                      )}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </section>

          {/* Table 3: Movement & Dynamics */}
          <section className="metrics-section">
            <div className="section-title-row">
              <h3>Movement &amp; Dynamics</h3>
              <span className="section-tag">Spatial Analysis</span>
            </div>
            <div className="table-wrapper">
              <table className="metrics-table">
                <thead>
                  <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Details</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td><strong>Total Displacement Events</strong></td>
                    <td className="metric-val">{metrics.movementEventsCount}</td>
                    <td>Number of times stationary objects shifted beyond minimum distance</td>
                  </tr>
                  <tr>
                    <td><strong>Most Active Object</strong></td>
                    <td className="metric-val">
                      {metrics.mostActiveObject ? metrics.mostActiveObject.label : "None"}
                    </td>
                    <td>
                      {metrics.mostActiveObject
                        ? `${metrics.mostActiveObject.moves} moves (${Math.round(metrics.mostActiveObject.distance)} px total distance)`
                        : "Scene was stationary"}
                    </td>
                  </tr>
                  <tr>
                    <td><strong>Scene Stability Ratio</strong></td>
                    <td className="metric-val">{metrics.stabilityRatio}</td>
                    <td>Overall stillness score throughout monitoring session</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </section>

          {/* Table 4: System & Pipeline Performance */}
          <section className="metrics-section">
            <div className="section-title-row">
              <h3>System &amp; Pipeline Performance</h3>
              <span className="section-tag">Technical Telemetry</span>
            </div>
            <div className="table-wrapper">
              <table className="metrics-table">
                <thead>
                  <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Benchmark / Context</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td><strong>Total Frames Analyzed</strong></td>
                    <td className="metric-val">{metrics.frameCount}</td>
                    <td>Frames sent and processed via FastAPI backend</td>
                  </tr>
                  <tr>
                    <td><strong>Average Processing Rate</strong></td>
                    <td className="metric-val">{metrics.avgFps}</td>
                    <td>Effective client-to-backend throughput</td>
                  </tr>
                  <tr>
                    <td><strong>Backend Model Inference</strong></td>
                    <td className="metric-val">{metrics.avgInferenceMs}</td>
                    <td>Average execution latency of YOLO11n on local backend</td>
                  </tr>
                  <tr>
                    <td><strong>Round-Trip Latency (RTT)</strong></td>
                    <td className="metric-val">{metrics.avgRttMs}</td>
                    <td>Average full cycle: frame extraction + upload + inference + response</td>
                  </tr>
                  <tr>
                    <td><strong>Applied Confidence Threshold</strong></td>
                    <td className="metric-val">{metrics.confidenceThreshold}</td>
                    <td>Filtering threshold used for bounding box detections</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </section>
        </div>
        </>
        )}

        {/* Email status banner if active */}
        {emailMessage && (
          <div
            style={{
              padding: "10px 24px",
              fontSize: "0.82rem",
              background:
                emailStatus === "success"
                  ? "rgba(16, 185, 129, 0.15)"
                  : "rgba(239, 68, 68, 0.15)",
              color: emailStatus === "success" ? "#34d399" : "#fca5a5",
              borderTop: "1px solid var(--border-subtle)",
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
            }}
          >
            <span>{emailMessage}</span>
            <button
              type="button"
              onClick={() => setEmailMessage("")}
              style={{
                background: "none",
                border: "none",
                color: "inherit",
                cursor: "pointer",
                fontWeight: "bold",
                fontSize: "1rem",
              }}
            >
              &times;
            </button>
          </div>
        )}

        {/* Modal Footer */}
        <div className="modal-footer">
          {viewMode === "summary" ? (
            <>
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => setViewMode("metrics")}
                title="Return to detailed metrics dashboard"
              >
                &larr; Back
              </button>

              <button
                type="button"
                className="btn btn-primary"
                onClick={handleCopy}
                title="Copy summary report to clipboard"
              >
                {copied ? "Copied to Clipboard!" : "Copy Summary"}
              </button>

              <button
                type="button"
                className="btn btn-secondary"
                onClick={onClose}
              >
                Done
              </button>
            </>
          ) : (
            <>
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => setViewMode("summary")}
                title="View executive session summary"
              >
                Show Summary
              </button>

              <button
                type="button"
                className="btn btn-primary"
                onClick={handleSendEmail}
                disabled={emailStatus === "sending"}
                title="Send metrics report to configured recipient"
              >
                {emailStatus === "sending"
                  ? "Sending Email..."
                  : emailStatus === "success"
                  ? "Email Sent"
                  : "Email Metrics"}
              </button>

              <button
                type="button"
                className="btn btn-secondary"
                onClick={onClose}
              >
                Done
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
