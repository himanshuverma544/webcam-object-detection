from __future__ import annotations

import html
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict

import httpx

from app.core.config import Settings

logger = logging.getLogger(__name__)


def build_metrics_html(metrics: Dict[str, Any], recipient: str) -> str:
    """Generate a responsive HTML email styled to match the Metrics Modal (no emojis)."""
    timestamp = html.escape(str(metrics.get("timestamp", "N/A")))
    duration_formatted = html.escape(str(metrics.get("durationFormatted", "N/A")))
    guarded_formatted = html.escape(str(metrics.get("guardedFormatted", "Not Armed")))
    frame_count = html.escape(str(metrics.get("frameCount", "0")))
    avg_fps = html.escape(str(metrics.get("avgFps", "N/A")))
    avg_inference = html.escape(str(metrics.get("avgInferenceMs", "N/A")))
    avg_rtt = html.escape(str(metrics.get("avgRttMs", "N/A")))
    confidence = html.escape(str(metrics.get("confidenceThreshold", "N/A")))
    peak_objects = html.escape(str(metrics.get("peakObjectCount", "0")))
    unique_entities = html.escape(str(metrics.get("uniqueEntitiesCount", "0")))

    # Security
    security_status = str(metrics.get("securityStatus", "CLEAN")).upper()
    status_message = html.escape(str(metrics.get("statusMessage", "No Incidents")))
    missing_incidents = html.escape(str(metrics.get("missingIncidentsCount", 0)))
    restored_count = html.escape(str(metrics.get("restoredCount", 0)))
    movement_count = html.escape(str(metrics.get("movementEventsCount", 0)))
    unresolved_count = int(metrics.get("unresolvedMissingCount", 0))

    unresolved_items = metrics.get("unresolvedMissing", [])
    if unresolved_items and isinstance(unresolved_items, list):
        missing_labels = ", ".join(
            html.escape(str(item.get("label", "item"))) for item in unresolved_items
        )
        unresolved_details = f"Still missing: {missing_labels}"
    else:
        unresolved_details = "None - all guarded items accounted for"

    # Status badge color
    if security_status == "ALERT":
        badge_bg = "rgba(239, 68, 68, 0.2)"
        badge_color = "#fca5a5"
        badge_border = "#ef4444"
        badge_prefix = "[ALERT]"
    elif security_status == "WARNING":
        badge_bg = "rgba(245, 158, 11, 0.2)"
        badge_color = "#fbbf24"
        badge_border = "#f59e0b"
        badge_prefix = "[WARNING]"
    else:
        badge_bg = "rgba(16, 185, 129, 0.2)"
        badge_color = "#34d399"
        badge_border = "#10b981"
        badge_prefix = "[CLEAN]"

    # Top classes
    top_classes = metrics.get("topClasses", [])
    if top_classes and isinstance(top_classes, list):
        class_items_html = " ".join(
            f'<span style="display:inline-block; margin:2px 4px; padding:3px 8px; background:#1e293b; color:#7dd3fc; border:1px solid #334155; border-radius:4px; font-size:12px;">'
            f'<strong>{html.escape(str(c.get("label", "")))}</strong> ({html.escape(str(c.get("percent", 0)))}% - {html.escape(str(c.get("count", 0)))} frames)</span>'
            for c in top_classes
        )
    else:
        class_items_html = '<span style="color:#64748b; font-size:13px;">No objects detected during session</span>'

    # Most active object
    most_active = metrics.get("mostActiveObject")
    if most_active and isinstance(most_active, dict):
        active_label = html.escape(str(most_active.get("label", "Object")))
        active_moves = html.escape(str(most_active.get("moves", 0)))
        active_dist = html.escape(str(round(float(most_active.get("distance", 0)))))
        active_details = f"{active_moves} moves ({active_dist} px total displacement)"
    else:
        active_label = "None"
        active_details = "Scene remained stationary"

    stability_ratio = html.escape(str(metrics.get("stabilityRatio", "100% Still")))

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Session Summary & Detection Metrics</title>
</head>
<body style="margin:0; padding:24px 12px; background-color:#0b0f19; font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; color:#f1f5f9; line-height:1.5;">
  <div style="max-width:760px; margin:0 auto; background-color:#131b2e; border:1px solid #26334d; border-radius:12px; overflow:hidden; box-shadow:0 12px 36px rgba(0,0,0,0.5);">
    
    <!-- Header -->
    <div style="padding:22px 24px 18px; border-bottom:1px solid #26334d; background-color:#0f172a;">
      <table style="width:100%; border-collapse:collapse;">
        <tr>
          <td style="vertical-align:middle;">
            <h1 style="margin:0; font-size:20px; font-weight:700; color:#f1f5f9; letter-spacing:-0.02em;">Session Summary &amp; Detection Metrics</h1>
            <p style="margin:4px 0 0; font-size:13px; color:#94a3b8;">
              Recorded: <strong>{timestamp}</strong> &bull; Total Session: <strong>{duration_formatted}</strong>
            </p>
          </td>
          <td style="text-align:right; vertical-align:middle;">
            <span style="display:inline-block; padding:6px 12px; background-color:{badge_bg}; color:{badge_color}; border:1px solid {badge_border}; border-radius:999px; font-size:12px; font-weight:600;">
              {badge_prefix} {status_message}
            </span>
          </td>
        </tr>
      </table>
    </div>

    <!-- KPI Grid -->
    <div style="padding:16px 24px; background-color:#0b0f19; border-bottom:1px solid #26334d;">
      <table style="width:100%; border-collapse:collapse;">
        <tr>
          <td style="width:25%; padding:6px 8px;">
            <div style="background:#131b2e; border:1px solid #26334d; border-radius:8px; padding:10px 12px;">
              <div style="font-size:10px; font-weight:600; text-transform:uppercase; color:#64748b; letter-spacing:0.04em;">Duration</div>
              <div style="font-size:17px; font-weight:700; color:#f1f5f9; margin-top:2px;">{duration_formatted}</div>
              <div style="font-size:11px; color:#94a3b8; margin-top:2px;">Guarded: {guarded_formatted}</div>
            </div>
          </td>
          <td style="width:25%; padding:6px 8px;">
            <div style="background:#131b2e; border:1px solid #26334d; border-radius:8px; padding:10px 12px;">
              <div style="font-size:10px; font-weight:600; text-transform:uppercase; color:#64748b; letter-spacing:0.04em;">Security Alerts</div>
              <div style="font-size:17px; font-weight:700; color:{'#f87171' if unresolved_count > 0 else '#f1f5f9'}; margin-top:2px;">{int(missing_incidents) + int(movement_count)}</div>
              <div style="font-size:11px; color:#94a3b8; margin-top:2px;">{unresolved_count} item(s) missing</div>
            </div>
          </td>
          <td style="width:25%; padding:6px 8px;">
            <div style="background:#131b2e; border:1px solid #26334d; border-radius:8px; padding:10px 12px;">
              <div style="font-size:10px; font-weight:600; text-transform:uppercase; color:#64748b; letter-spacing:0.04em;">Entities</div>
              <div style="font-size:17px; font-weight:700; color:#f1f5f9; margin-top:2px;">{unique_entities}</div>
              <div style="font-size:11px; color:#94a3b8; margin-top:2px;">Peak in frame: {peak_objects}</div>
            </div>
          </td>
          <td style="width:25%; padding:6px 8px;">
            <div style="background:#131b2e; border:1px solid #26334d; border-radius:8px; padding:10px 12px;">
              <div style="font-size:10px; font-weight:600; text-transform:uppercase; color:#64748b; letter-spacing:0.04em;">Backend Latency</div>
              <div style="font-size:17px; font-weight:700; color:#38bdf8; margin-top:2px;">{avg_inference}</div>
              <div style="font-size:11px; color:#94a3b8; margin-top:2px;">Rate: {avg_fps}</div>
            </div>
          </td>
        </tr>
      </table>
    </div>

    <!-- Tables Container -->
    <div style="padding:20px 24px;">

      <!-- Table 1: Security & Incident Alerts -->
      <div style="margin-bottom:22px;">
        <div style="margin-bottom:8px;">
          <strong style="font-size:14px; color:#f1f5f9;">Security &amp; Incident Alerts</strong>
          <span style="font-size:10px; text-transform:uppercase; background:#1e293b; color:#94a3b8; padding:2px 6px; border-radius:4px; margin-left:8px; border:1px solid #334155;">Surveillance &amp; Guard</span>
        </div>
        <table style="width:100%; border-collapse:collapse; background:#1a2540; border:1px solid #26334d; border-radius:8px; overflow:hidden; font-size:13px;">
          <thead>
            <tr style="background:#0b0f19; text-align:left;">
              <th style="padding:8px 12px; color:#64748b; font-size:11px; text-transform:uppercase; border-bottom:1px solid #26334d;">Metric</th>
              <th style="padding:8px 12px; color:#64748b; font-size:11px; text-transform:uppercase; border-bottom:1px solid #26334d;">Value</th>
              <th style="padding:8px 12px; color:#64748b; font-size:11px; text-transform:uppercase; border-bottom:1px solid #26334d;">Status &amp; Details</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td style="padding:8px 12px; color:#f1f5f9; border-bottom:1px solid #26334d;"><strong>Missing Item Incidents</strong></td>
              <td style="padding:8px 12px; color:#f1f5f9; font-weight:600; border-bottom:1px solid #26334d;">{missing_incidents}</td>
              <td style="padding:8px 12px; color:#94a3b8; border-bottom:1px solid #26334d;">Total registered item disappearances</td>
            </tr>
            <tr style="{'background:rgba(239,68,68,0.1);' if unresolved_count > 0 else ''}">
              <td style="padding:8px 12px; color:#f1f5f9; border-bottom:1px solid #26334d;"><strong>Unresolved Missing Items</strong></td>
              <td style="padding:8px 12px; color:{'#f87171' if unresolved_count > 0 else '#f1f5f9'}; font-weight:700; border-bottom:1px solid #26334d;">{unresolved_count}</td>
              <td style="padding:8px 12px; color:{'#f87171' if unresolved_count > 0 else '#94a3b8'}; border-bottom:1px solid #26334d;">{unresolved_details}</td>
            </tr>
            <tr>
              <td style="padding:8px 12px; color:#f1f5f9; border-bottom:1px solid #26334d;"><strong>Restored Items</strong></td>
              <td style="padding:8px 12px; color:#f1f5f9; font-weight:600; border-bottom:1px solid #26334d;">{restored_count}</td>
              <td style="padding:8px 12px; color:#94a3b8; border-bottom:1px solid #26334d;">Previously missing items returned to view</td>
            </tr>
            <tr>
              <td style="padding:8px 12px; color:#f1f5f9; border-bottom:1px solid #26334d;"><strong>Movement &amp; Tamper Alerts</strong></td>
              <td style="padding:8px 12px; color:#f1f5f9; font-weight:600; border-bottom:1px solid #26334d;">{movement_count}</td>
              <td style="padding:8px 12px; color:#94a3b8; border-bottom:1px solid #26334d;">Displacement events exceeding distance threshold</td>
            </tr>
            <tr>
              <td style="padding:8px 12px; color:#f1f5f9;"><strong>Scene Guard Protection</strong></td>
              <td style="padding:8px 12px; color:#f1f5f9; font-weight:600;">{guarded_formatted}</td>
              <td style="padding:8px 12px; color:#94a3b8;">Active scene surveillance baseline</td>
            </tr>
          </tbody>
        </table>
      </div>

      <!-- Table 2: Activity & Object Analytics -->
      <div style="margin-bottom:22px;">
        <div style="margin-bottom:8px;">
          <strong style="font-size:14px; color:#f1f5f9;">Activity &amp; Object Analytics</strong>
          <span style="font-size:10px; text-transform:uppercase; background:#1e293b; color:#94a3b8; padding:2px 6px; border-radius:4px; margin-left:8px; border:1px solid #334155;">Detection Insights</span>
        </div>
        <table style="width:100%; border-collapse:collapse; background:#1a2540; border:1px solid #26334d; border-radius:8px; overflow:hidden; font-size:13px;">
          <thead>
            <tr style="background:#0b0f19; text-align:left;">
              <th style="padding:8px 12px; color:#64748b; font-size:11px; text-transform:uppercase; border-bottom:1px solid #26334d;">Metric</th>
              <th style="padding:8px 12px; color:#64748b; font-size:11px; text-transform:uppercase; border-bottom:1px solid #26334d;">Value</th>
              <th style="padding:8px 12px; color:#64748b; font-size:11px; text-transform:uppercase; border-bottom:1px solid #26334d;">Details &amp; Breakdown</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td style="padding:8px 12px; color:#f1f5f9; border-bottom:1px solid #26334d;"><strong>Session Active Duration</strong></td>
              <td style="padding:8px 12px; color:#f1f5f9; font-weight:600; border-bottom:1px solid #26334d;">{duration_formatted}</td>
              <td style="padding:8px 12px; color:#94a3b8; border-bottom:1px solid #26334d;">Continuous webcam monitoring time</td>
            </tr>
            <tr>
              <td style="padding:8px 12px; color:#f1f5f9; border-bottom:1px solid #26334d;"><strong>Peak Simultaneous Objects</strong></td>
              <td style="padding:8px 12px; color:#f1f5f9; font-weight:600; border-bottom:1px solid #26334d;">{peak_objects}</td>
              <td style="padding:8px 12px; color:#94a3b8; border-bottom:1px solid #26334d;">Highest number of concurrent detections in a frame</td>
            </tr>
            <tr>
              <td style="padding:8px 12px; color:#f1f5f9; border-bottom:1px solid #26334d;"><strong>Total Tracked Entities</strong></td>
              <td style="padding:8px 12px; color:#f1f5f9; font-weight:600; border-bottom:1px solid #26334d;">{unique_entities}</td>
              <td style="padding:8px 12px; color:#94a3b8; border-bottom:1px solid #26334d;">Distinct object tracks registered</td>
            </tr>
            <tr>
              <td style="padding:8px 12px; color:#f1f5f9;"><strong>Top Detected Classes</strong></td>
              <td colspan="2" style="padding:8px 12px;">{class_items_html}</td>
            </tr>
          </tbody>
        </table>
      </div>

      <!-- Table 3: Movement & Dynamics -->
      <div style="margin-bottom:22px;">
        <div style="margin-bottom:8px;">
          <strong style="font-size:14px; color:#f1f5f9;">Movement &amp; Dynamics</strong>
          <span style="font-size:10px; text-transform:uppercase; background:#1e293b; color:#94a3b8; padding:2px 6px; border-radius:4px; margin-left:8px; border:1px solid #334155;">Spatial Analysis</span>
        </div>
        <table style="width:100%; border-collapse:collapse; background:#1a2540; border:1px solid #26334d; border-radius:8px; overflow:hidden; font-size:13px;">
          <thead>
            <tr style="background:#0b0f19; text-align:left;">
              <th style="padding:8px 12px; color:#64748b; font-size:11px; text-transform:uppercase; border-bottom:1px solid #26334d;">Metric</th>
              <th style="padding:8px 12px; color:#64748b; font-size:11px; text-transform:uppercase; border-bottom:1px solid #26334d;">Value</th>
              <th style="padding:8px 12px; color:#64748b; font-size:11px; text-transform:uppercase; border-bottom:1px solid #26334d;">Details</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td style="padding:8px 12px; color:#f1f5f9; border-bottom:1px solid #26334d;"><strong>Total Displacement Events</strong></td>
              <td style="padding:8px 12px; color:#f1f5f9; font-weight:600; border-bottom:1px solid #26334d;">{movement_count}</td>
              <td style="padding:8px 12px; color:#94a3b8; border-bottom:1px solid #26334d;">Number of times stationary objects shifted</td>
            </tr>
            <tr>
              <td style="padding:8px 12px; color:#f1f5f9; border-bottom:1px solid #26334d;"><strong>Most Active Object</strong></td>
              <td style="padding:8px 12px; color:#f1f5f9; font-weight:600; border-bottom:1px solid #26334d;">{active_label}</td>
              <td style="padding:8px 12px; color:#94a3b8; border-bottom:1px solid #26334d;">{active_details}</td>
            </tr>
            <tr>
              <td style="padding:8px 12px; color:#f1f5f9;"><strong>Scene Stability Ratio</strong></td>
              <td style="padding:8px 12px; color:#f1f5f9; font-weight:600;">{stability_ratio}</td>
              <td style="padding:8px 12px; color:#94a3b8;">Overall stationary time ratio</td>
            </tr>
          </tbody>
        </table>
      </div>

      <!-- Table 4: System & Pipeline Performance -->
      <div>
        <div style="margin-bottom:8px;">
          <strong style="font-size:14px; color:#f1f5f9;">System &amp; Pipeline Performance</strong>
          <span style="font-size:10px; text-transform:uppercase; background:#1e293b; color:#94a3b8; padding:2px 6px; border-radius:4px; margin-left:8px; border:1px solid #334155;">Technical Telemetry</span>
        </div>
        <table style="width:100%; border-collapse:collapse; background:#1a2540; border:1px solid #26334d; border-radius:8px; overflow:hidden; font-size:13px;">
          <thead>
            <tr style="background:#0b0f19; text-align:left;">
              <th style="padding:8px 12px; color:#64748b; font-size:11px; text-transform:uppercase; border-bottom:1px solid #26334d;">Metric</th>
              <th style="padding:8px 12px; color:#64748b; font-size:11px; text-transform:uppercase; border-bottom:1px solid #26334d;">Value</th>
              <th style="padding:8px 12px; color:#64748b; font-size:11px; text-transform:uppercase; border-bottom:1px solid #26334d;">Benchmark / Context</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td style="padding:8px 12px; color:#f1f5f9; border-bottom:1px solid #26334d;"><strong>Total Frames Analyzed</strong></td>
              <td style="padding:8px 12px; color:#f1f5f9; font-weight:600; border-bottom:1px solid #26334d;">{frame_count}</td>
              <td style="padding:8px 12px; color:#94a3b8; border-bottom:1px solid #26334d;">Frames sent and processed via FastAPI backend</td>
            </tr>
            <tr>
              <td style="padding:8px 12px; color:#f1f5f9; border-bottom:1px solid #26334d;"><strong>Average Processing Rate</strong></td>
              <td style="padding:8px 12px; color:#f1f5f9; font-weight:600; border-bottom:1px solid #26334d;">{avg_fps}</td>
              <td style="padding:8px 12px; color:#94a3b8; border-bottom:1px solid #26334d;">Effective client-to-backend throughput</td>
            </tr>
            <tr>
              <td style="padding:8px 12px; color:#f1f5f9; border-bottom:1px solid #26334d;"><strong>Backend Model Inference</strong></td>
              <td style="padding:8px 12px; color:#f1f5f9; font-weight:600; border-bottom:1px solid #26334d;">{avg_inference}</td>
              <td style="padding:8px 12px; color:#94a3b8; border-bottom:1px solid #26334d;">Average execution latency of YOLO11n on local backend</td>
            </tr>
            <tr>
              <td style="padding:8px 12px; color:#f1f5f9; border-bottom:1px solid #26334d;"><strong>Round-Trip Latency (RTT)</strong></td>
              <td style="padding:8px 12px; color:#f1f5f9; font-weight:600; border-bottom:1px solid #26334d;">{avg_rtt}</td>
              <td style="padding:8px 12px; color:#94a3b8; border-bottom:1px solid #26334d;">Average full cycle: extraction + upload + inference + response</td>
            </tr>
            <tr>
              <td style="padding:8px 12px; color:#f1f5f9;"><strong>Applied Confidence Threshold</strong></td>
              <td style="padding:8px 12px; color:#f1f5f9; font-weight:600;">{confidence}</td>
              <td style="padding:8px 12px; color:#94a3b8;">Filtering threshold used for bounding box detections</td>
            </tr>
          </tbody>
        </table>
      </div>

    </div>

    <!-- Footer -->
    <div style="padding:14px 24px; border-top:1px solid #26334d; background-color:#0f172a; text-align:center; font-size:12px; color:#64748b;">
      Dispatched to {html.escape(recipient)} &bull; Generated by Webcam Object Detection (YOLO11n + FastAPI)
    </div>

  </div>
</body>
</html>
"""
    return html_content


def build_metrics_text(metrics: Dict[str, Any]) -> str:
    """Generate a clean plain-text version of session metrics."""
    return f"""
WEBCAM OBJECT DETECTION SESSION REPORT
Timestamp: {metrics.get('timestamp')}
Duration: {metrics.get('durationFormatted')}
Guarded: {metrics.get('guardedFormatted')}
Frames: {metrics.get('frameCount')} ({metrics.get('avgFps')})

SECURITY & ALERTS
- Status: {metrics.get('statusMessage')}
- Missing Incidents: {metrics.get('missingIncidentsCount')}
- Unresolved Missing: {metrics.get('unresolvedMissingCount')}
- Restored Items: {metrics.get('restoredCount')}
- Movement Alerts: {metrics.get('movementEventsCount')}

ACTIVITY & OBJECTS
- Peak Objects: {metrics.get('peakObjectCount')}
- Unique Entities: {metrics.get('uniqueEntitiesCount')}

MOVEMENT & DYNAMICS
- Total Movement Events: {metrics.get('movementEventsCount')}
- Scene Stability: {metrics.get('stabilityRatio')}

SYSTEM PERFORMANCE
- Avg Backend Inference: {metrics.get('avgInferenceMs')}
- Avg Round-Trip Latency: {metrics.get('avgRttMs')}
- Confidence: {metrics.get('confidenceThreshold')}
""".strip()


def send_via_resend(
    recipient: str,
    metrics: Dict[str, Any],
    settings: Settings,
) -> None:
    """Dispatch structured metrics email via Resend REST API."""
    subject = f"Detection Session Metrics Report - {metrics.get('timestamp', 'Session')}"
    html_content = build_metrics_html(metrics, recipient)
    text_content = build_metrics_text(metrics)

    headers = {
        "Authorization": f"Bearer {settings.resend_api_key.strip()}",
        "Content-Type": "application/json",
    }
    payload = {
        "from": settings.resend_from_email,
        "to": [recipient],
        "subject": subject,
        "html": html_content,
        "text": text_content,
    }

    logger.info("Dispatching metrics email to %s via Resend API", recipient)
    try:
        with httpx.Client(timeout=15.0) as client:
            response = client.post(
                "https://api.resend.com/emails",
                json=payload,
                headers=headers,
            )

        if response.status_code >= 400:
            error_message = response.text
            try:
                data = response.json()
                error_message = data.get("message") or data.get("error") or error_message
            except Exception:
                pass
            logger.error("Resend API rejected request (%d): %s", response.status_code, error_message)
            raise ValueError(f"Resend API error: {error_message}")

        logger.info("Metrics email successfully dispatched via Resend to %s", recipient)
    except httpx.RequestError as exc:
        logger.error("Network error while connecting to Resend API: %s", exc)
        raise ValueError(f"Failed to connect to Resend API: {exc}") from exc


def send_via_smtp(
    recipient: str,
    metrics: Dict[str, Any],
    settings: Settings,
) -> None:
    """Send structured metrics report email via SMTP fallback."""
    smtp_user = settings.smtp_user
    smtp_password = settings.smtp_password

    if not smtp_user or not smtp_password:
        raise ValueError(
            "Neither RESEND_API_KEY nor SMTP credentials are configured. "
            "Please configure RESEND_API_KEY in backend/.env to send emails."
        )

    sender_email = settings.smtp_from_email or smtp_user
    subject = f"Detection Session Metrics Report - {metrics.get('timestamp', 'Session')}"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = recipient

    text_content = build_metrics_text(metrics)
    html_content = build_metrics_html(metrics, recipient)

    msg.attach(MIMEText(text_content, "plain", "utf-8"))
    msg.attach(MIMEText(html_content, "html", "utf-8"))

    logger.info("Connecting to SMTP host %s:%d to send email to %s", settings.smtp_host, settings.smtp_port, recipient)

    try:
        if settings.smtp_port == 465:
            server = smtplib.SMTP_SSL(settings.smtp_host, settings.smtp_port, timeout=15)
        else:
            server = smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=15)
            if settings.smtp_tls:
                server.starttls()

        server.login(smtp_user, smtp_password)
        server.sendmail(sender_email, [recipient], msg.as_string())
        server.quit()
        logger.info("Metrics email successfully sent via SMTP to %s", recipient)
    except smtplib.SMTPAuthenticationError as err:
        logger.error("SMTP Authentication Failed: %s", err)
        raise ValueError("SMTP authentication failed. Please verify your email and app password.") from err
    except Exception as err:
        logger.error("Failed to deliver metrics email via SMTP: %s", err)
        raise ValueError(f"Failed to send email: {err}") from err


def send_metrics_email(
    recipient: str,
    metrics: Dict[str, Any],
    settings: Settings,
) -> None:
    """Send structured metrics report email using Resend API or SMTP fallback."""
    if settings.resend_api_key and settings.resend_api_key.strip():
        send_via_resend(recipient=recipient, metrics=metrics, settings=settings)
    elif settings.smtp_user and settings.smtp_password:
        send_via_smtp(recipient=recipient, metrics=metrics, settings=settings)
    else:
        raise ValueError(
            "Email service is not configured. Please set RESEND_API_KEY in backend/.env to send emails."
        )

