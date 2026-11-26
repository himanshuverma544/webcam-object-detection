from __future__ import annotations

import logging
from typing import Any, Dict, Optional
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Predefined trial templates supported by Twilio trial accounts
TRIAL_TEMPLATES = {
    "alerts": "sms_account_alerts",
    "event": "sms_event_notifications",
    "internal": "sms_internal_alerts",
    "reminder": "sms_appointment_reminders",
    "delivery": "sms_delivery_updates",
}


def get_twilio_client(settings: Settings) -> Client:
    """Instantiate a Twilio REST Client using configured credentials."""
    if not settings.twilio_account_sid or not settings.twilio_auth_token:
        raise ValueError(
            "Twilio credentials missing. Please set TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN in backend/.env"
        )
    return Client(settings.twilio_account_sid, settings.twilio_auth_token)


def send_sms(
    message: str,
    recipient: Optional[str] = None,
    settings: Optional[Settings] = None,
    fallback_trial_template: str = "sms_event_notifications",
) -> Dict[str, Any]:
    """
    Send an SMS message via Twilio API.
    
    If the account is a Twilio Trial account (which restricts custom text bodies and enforces
    error 572006), automatically falls back to an approved predefined trial template.

    :param message: The text message content to send (or template key if on trial).
    :param recipient: Recipient phone number in E.164 format (e.g. +919410986033).
    :param settings: Application settings.
    :param fallback_trial_template: Predefined template to use if trial mode rejects custom text.
    :return: Dictionary containing status, message SID, and recipient.
    """
    if settings is None:
        settings = get_settings()

    target_number = (recipient or settings.recipient_phone_number or "").strip()
    if not target_number:
        raise ValueError(
            "No recipient phone number specified. Provide recipient or set RECIPIENT_PHONE_NUMBER in backend/.env"
        )

    if not settings.twilio_phone_number:
        raise ValueError(
            "No Twilio sender phone number configured. Set TWILIO_PHONE_NUMBER in backend/.env"
        )

    client = get_twilio_client(settings)

    # 1. Attempt sending the message body directly
    try:
        logger.info("Dispatching SMS via Twilio to %s...", target_number)
        response = client.messages.create(
            body=message,
            from_=settings.twilio_phone_number,
            to=target_number,
        )
        logger.info(
            "SMS successfully dispatched via Twilio. SID: %s, Status: %s",
            response.sid,
            response.status,
        )
        return {
            "status": "sent",
            "sid": response.sid,
            "recipient": target_number,
            "delivery_status": response.status,
            "mode": "custom",
        }
    except TwilioRestException as exc:
        # Check if error is 572006 (Twilio Free Trial restriction: predefined templates only)
        if exc.code == 572006:
            logger.warning(
                "Twilio Trial restriction (code 572006): Free trial accounts can only send predefined templates. "
                "Falling back to template '%s'. To send custom summaries, upgrade your Twilio account.",
                fallback_trial_template,
            )
            try:
                trial_response = client.messages.create(
                    body=fallback_trial_template,
                    from_=settings.twilio_phone_number,
                    to=target_number,
                )
                return {
                    "status": "sent",
                    "sid": trial_response.sid,
                    "recipient": target_number,
                    "delivery_status": trial_response.status,
                    "mode": "trial_template_fallback",
                    "template_used": fallback_trial_template,
                    "notice": (
                        "Twilio Free Trial accounts only allow predefined templates. "
                        "The test SMS was dispatched using an approved template. "
                        "Upgrade Twilio to enable custom text/summary bodies."
                    ),
                }
            except Exception as trial_exc:
                logger.error("Failed to send even trial template: %s", trial_exc)
                raise ValueError(f"Twilio trial template error: {trial_exc}") from trial_exc

        logger.error("Twilio API error while sending SMS to %s: %s (code %s)", target_number, exc.msg, exc.code)
        raise ValueError(f"Twilio error ({exc.code}): {exc.msg}") from exc
    except Exception as exc:
        logger.exception("Unexpected error while dispatching SMS via Twilio.")
        raise ValueError(f"Failed to send SMS: {exc}") from exc


def send_greeting_sms(
    recipient: Optional[str] = None,
    settings: Optional[Settings] = None,
) -> Dict[str, Any]:
    """
    Send a friendly greeting SMS to verify Twilio integration.
    """
    greeting_text = (
        "Hello from your Webcam Object Detection App! 👋\n"
        "Twilio SMS integration is working successfully."
    )
    return send_sms(
        message=greeting_text,
        recipient=recipient,
        settings=settings,
        fallback_trial_template="sms_event_notifications",
    )


def build_metrics_sms_text(metrics: Dict[str, Any]) -> str:
    """Format session metrics into a compact SMS-friendly summary (under 320 chars)."""
    status_msg = str(metrics.get("statusMessage") or metrics.get("securityStatus") or "Normal").strip()
    duration = str(metrics.get("durationFormatted") or "N/A").strip()
    frames = str(metrics.get("frameCount") or 0)
    peak_objs = str(metrics.get("peakObjectCount") or 0)
    unique_entities = str(metrics.get("uniqueEntitiesCount") or 0)
    missing = int(metrics.get("missingIncidentsCount") or 0)
    moves = int(metrics.get("movementEventsCount") or 0)

    lines = [
        "[Webcam Detection Summary]",
        f"Status: {status_msg}",
        f"Duration: {duration} | Frames: {frames}",
        f"Objects: Peak {peak_objs}, Unique {unique_entities}",
        f"Security: {missing} missing, {moves} movements",
    ]
    return "\n".join(lines)


def send_metrics_sms(
    metrics: Dict[str, Any],
    recipient: Optional[str] = None,
    settings: Optional[Settings] = None,
) -> Dict[str, Any]:
    """
    Format detection session metrics and dispatch via SMS.
    """
    summary_text = build_metrics_sms_text(metrics)
    result = send_sms(
        message=summary_text,
        recipient=recipient,
        settings=settings,
        fallback_trial_template="sms_event_notifications",
    )
    result["summary_preview"] = summary_text
    return result
