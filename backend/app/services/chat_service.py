from __future__ import annotations

import base64
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import httpx

from app.core.config import Settings
from app.schemas.chat import ChatAction, ChatMessage, ChatResponse

logger = logging.getLogger(__name__)

ALL_COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

GEMINI_TOOLS_DECLARATION = [
    {
        "name": "start_camera",
        "description": "Start or activate the webcam video stream.",
        "parameters": {
            "type": "OBJECT",
            "properties": {},
        },
    },
    {
        "name": "stop_camera",
        "description": "Stop or deactivate the webcam video stream and snapshot session metrics.",
        "parameters": {
            "type": "OBJECT",
            "properties": {},
        },
    },
    {
        "name": "toggle_detection",
        "description": "Pause or resume real-time object detection inference loop.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {
                    "type": "STRING",
                    "enum": ["pause", "resume", "toggle"],
                    "description": "Whether to pause, resume, or toggle detection state.",
                }
            },
            "required": ["action"],
        },
    },
    {
        "name": "set_confidence",
        "description": "Update the object detection confidence threshold (between 0.01 and 1.0, e.g. 0.50).",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "threshold": {
                    "type": "NUMBER",
                    "description": "Confidence threshold float between 0.01 and 1.0 (e.g. 0.65 for 65%).",
                }
            },
            "required": ["threshold"],
        },
    },
    {
        "name": "set_frame_cadence",
        "description": "Set the webcam frame capture interval / cadence in milliseconds (200, 250, 300, or 500 ms).",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "interval_ms": {
                    "type": "INTEGER",
                    "description": "Interval in milliseconds: 200 (~5 FPS), 250 (4 FPS), 300 (~3.3 FPS), or 500 (2 FPS).",
                }
            },
            "required": ["interval_ms"],
        },
    },
    {
        "name": "set_min_move_distance",
        "description": "Set minimum movement distance threshold in pixels (25 to 150 px) for scene movement tracking.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "distance_px": {
                    "type": "INTEGER",
                    "description": "Pixel distance threshold (25 - 150).",
                }
            },
            "required": ["distance_px"],
        },
    },
    {
        "name": "set_settle_dwell_time",
        "description": "Set settle dwell duration in milliseconds (800 to 3000 ms) before confirming an object has finished moving.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "dwell_ms": {
                    "type": "INTEGER",
                    "description": "Dwell time in milliseconds (e.g. 1500 for 1.5 seconds).",
                }
            },
            "required": ["dwell_ms"],
        },
    },
    {
        "name": "set_coco_classes",
        "description": "Filter or update the active COCO object detection classes.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "mode": {
                    "type": "STRING",
                    "enum": ["replace", "add", "remove", "select_all", "clear_all"],
                    "description": "How to update classes: 'replace' with a given list, 'add' classes, 'remove' classes, 'select_all', or 'clear_all'.",
                },
                "classes": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"},
                    "description": "List of COCO class names (e.g. ['person', 'cup', 'laptop']).",
                },
            },
            "required": ["mode"],
        },
    },
    {
        "name": "toggle_custom_item",
        "description": "Enable or disable detection for a specific custom visual reference item.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "item_name_or_id": {
                    "type": "STRING",
                    "description": "Name or ID of the custom item.",
                },
                "enable": {
                    "type": "BOOLEAN",
                    "description": "True to enable detection, False to disable.",
                },
            },
            "required": ["item_name_or_id", "enable"],
        },
    },
    {
        "name": "delete_custom_item",
        "description": "Delete or remove a custom visual reference item (e.g. 'delete My Item', 'remove Coffee Mug', 'delete custom item Backpack').",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "item_name_or_id": {
                    "type": "STRING",
                    "description": "Name or ID of the custom item to delete.",
                }
            },
            "required": ["item_name_or_id"],
        },
    },
    {
        "name": "add_images_to_custom_item",
        "description": "Add more reference photos/snapshots to an existing custom item to improve its detection accuracy.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "item_name_or_id": {
                    "type": "STRING",
                    "description": "Name or ID of the existing custom item to update with additional photos.",
                }
            },
            "required": ["item_name_or_id"],
        },
    },
    {
        "name": "set_similarity_threshold",
        "description": "Set the Match Strictness / Visual Similarity threshold for custom items (0.1 to 1.0, e.g. 0.75 for 75%). Call this whenever the user asks to change or set 'match strictness', 'strictness', or 'similarity'.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "threshold": {
                    "type": "NUMBER",
                    "description": "Match strictness threshold between 0.1 and 1.0.",
                }
            },
            "required": ["threshold"],
        },
    },
    {
        "name": "capture_snapshot_as_custom_item",
        "description": "Register and save a custom visual item using either staged reference photos/snapshots or by capturing the live webcam frame. Item name must be unique.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "item_name": {
                    "type": "STRING",
                    "description": "Display name for the custom item (e.g. 'My Item', 'Coffee Mug', 'Backpack'). Must not duplicate any existing custom item name.",
                }
            },
            "required": ["item_name"],
        },
    },
    {
        "name": "guard_scene",
        "description": "Arm or disarm scene guarding. When armed, currently detected objects are locked as baseline, and missing alerts trigger if any baseline item disappears.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "arm": {
                    "type": "BOOLEAN",
                    "description": "True to arm/guard the scene, False to disarm.",
                }
            },
            "required": ["arm"],
        },
    },
    {
        "name": "dismiss_missing_alert",
        "description": "Dismiss an active missing item alert banner by item name or ID.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "item_label": {
                    "type": "STRING",
                    "description": "Label or class name of the missing item to dismiss.",
                }
            },
            "required": ["item_label"],
        },
    },
    {
        "name": "show_metrics",
        "description": "Open or switch the session metrics and executive summary modal dialog. Pass tab='summary' to show the executive summary text report, or tab='metrics' to show the telemetry grid and detection stats.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "tab": {
                    "type": "STRING",
                    "enum": ["metrics", "summary"],
                    "description": "The view mode to display: 'summary' for executive markdown session report, or 'metrics' for telemetry metrics grid. Defaults to 'summary' when user asks for the summary or report.",
                }
            },
        },
    },
    {
        "name": "send_metrics_email",
        "description": "Send structured session metrics report to an email address.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "recipient_email": {
                    "type": "STRING",
                    "description": "Recipient email address (optional, uses default if omitted).",
                }
            },
        },
    },
    {
        "name": "copy_summary",
        "description": "Copy the executive session summary report text to the user's clipboard. Call this whenever the user asks to 'copy summary', 'copy executive summary', or 'copy report'.",
        "parameters": {
            "type": "OBJECT",
            "properties": {},
        },
    },
    {
        "name": "close_metrics_modal",
        "description": "Close or dismiss the session metrics and executive summary modal dialog on the UI. Call this whenever the user asks to close, dismiss, hide, or finish viewing the modal or summary (e.g. 'close the summary', 'close modal', 'done', 'dismiss metrics').",
        "parameters": {
            "type": "OBJECT",
            "properties": {},
        },
    },
    {
        "name": "reset_tracker",
        "description": "Reset tracker state, movement logs, and object tracking IDs.",
        "parameters": {
            "type": "OBJECT",
            "properties": {},
        },
    },
]

SYSTEM_PROMPT = """You are the Vision AI Copilot, the dedicated in-app assistant and controller for this Real-Time Webcam Object Detection & Scene Security Dashboard.

CRITICAL BOUNDARIES & DOMAIN SCOPE:
1. You exist SOLELY to control, monitor, configure, and explain THIS application (YOLO11 object detection, custom visual items via YOLOE, webcam video feed, scene guarding, telemetry metrics, and email reports).
2. DO NOT answer general programming questions, coding tutorials, or computer science concepts (for example: "what is useEffect", "how to write a loop in Python", "explain redux", "write HTML/CSS").
3. DO NOT answer general trivia, creative writing requests, history, or off-topic queries unrelated to this object detection dashboard.
4. If a user asks an out-of-scope or unrelated question, POLITELY DECLINE and redirect them back to the dashboard controls.
   Example refusal: "I'm your Vision AI Copilot, dedicated exclusively to controlling and monitoring this webcam object detection and scene security dashboard. I can help you start/stop the camera, select detection classes (like person or bottle), guard the scene against theft, adjust confidence, or save custom items. What would you like to configure on the dashboard?"

DASHBOARD EXPLANATIONS:
If asked about dashboard settings (e.g. "What is dwell time?", "What is scene guarding?", "What classes can you detect?", "How do custom items work?"):
Explain strictly in the context of this specific dashboard's computer vision and tracking features.

CAPABILITIES & TOOL USAGE:
Call the provided tools whenever the user prompts to change settings or perform actions:
- Camera: start_camera, stop_camera
- Detection inference: toggle_detection
- COCO Classes: set_coco_classes (modes: select_all, clear_all, add, remove, replace)
- Custom Visual Items:
  * capture_snapshot_as_custom_item: Registers a new custom item using the live camera frame OR using attached reference images. Call this whenever the user wants to save or register a custom item.
    - Always extract the exact item_name specified by the user (e.g. for "Save these 2 reference photos as custom item 'My Item'", item_name is "My Item").
    - If Camera is NOT 'active' and no image is attached (AttachedImages=false), DO NOT call capture_snapshot_as_custom_item. Instead explain:
      "The camera is currently stopped. Please start the camera first or attach reference photos so I can register your item."
    - If the user did not specify an item name, ask them:
      "Please provide a name for your custom item (e.g. 'take snapshot and save as Coffee Mug')."
    - If a custom item with the same name already exists in CustomItems (case-insensitive):
      DO NOT call capture_snapshot_as_custom_item. Instead, inform the user:
      "A custom item named '{item_name}' already exists. Please choose a unique name or delete the existing item first."
  * toggle_custom_item: Enable or disable detection for a custom item.
  * delete_custom_item: Call this whenever the user asks to delete or remove a custom item (e.g. 'delete My Item', 'remove Coffee Mug', 'delete custom item Backpack'). Match against CustomItems.
  * add_images_to_custom_item: Call this when the user wants to add or append more reference photos to an existing custom item (e.g. 'add these photos to My Item', 'add more images to Coffee Mug').
  * set_similarity_threshold: Set the Match Strictness / Similarity threshold for custom visual items (0.10 to 1.0, e.g. 0.75 for 75%). Call this whenever the user asks to set or change "match strictness", "strictness", "match threshold", or "similarity".
    - NOTE: Do NOT call set_confidence when the user asks to change Match Strictness or Strictness! Only call set_confidence for YOLO/COCO confidence threshold.
- Sliders:
  * set_confidence (0.01 to 1.0): COCO detection confidence threshold only. DO NOT call this when user asks for Match Strictness.
  * set_frame_cadence (200, 250, 300, 500 ms)
  * set_min_move_distance (25 to 150 px)
  * set_settle_dwell_time (800 to 3000 ms)
- Security & Alerts:
  * guard_scene (arm baseline / disarm)
  * dismiss_missing_alert
- Metrics & Reporting:
  * CRITICAL TIMING RULE: Session metrics and the executive summary are recorded and finalized ONLY after stopping the camera (when Camera is NOT 'active', or hasMetrics is true).
  * If the camera is currently active (Camera=active) and the user asks to show or view metrics/summary without asking to stop the camera:
    DO NOT call stop_camera and DO NOT call show_metrics. You MUST NOT stop the camera unless the user explicitly asks to stop it or confirms "yes". Instead, reply with text only:
    "Session metrics and the executive summary will be available after stopping the camera. Would you like me to stop the camera and show your session summary now?"
  * If the camera is currently active (Camera=active) and the user asks to email or send metrics without asking to stop the camera:
    DO NOT call stop_camera and DO NOT call send_metrics_email. You MUST NOT stop the camera unless the user explicitly asks to stop it or confirms "yes". Instead, reply with text only:
    "I cannot send a metrics report yet because the camera is currently active and no session data has been finalized. Session metrics are generated only after you stop the camera. Would you like me to stop the camera now so we can finalize the session and send the report?"
  * If the user explicitly asks to "stop camera and show summary" (or confirms "yes", "sure", "ok" to stopping the camera and viewing the summary):
    Call stop_camera and show_metrics(tab='summary') together.
  * If the user explicitly asks to "stop camera and show metrics" (or confirms "yes", "sure", "ok" to stopping the camera and viewing metrics):
    Call stop_camera and show_metrics(tab='metrics') together.
  * If the user explicitly asks to "stop camera and send metrics" (or confirms "yes", "sure", "ok" to stopping the camera and sending the report):
    Call stop_camera and send_metrics_email together.
  * If the user explicitly asks to "stop camera and copy summary" (or confirms "yes", "sure", "ok" to stopping the camera and copying the summary):
    Call stop_camera and copy_summary together.
  * If the camera is stopped and hasMetrics is true:
    - User asks to "show summary" / "view summary" / "executive summary" / "report" -> call show_metrics(tab='summary').
    - User asks to "show metrics" / "open metrics" / "telemetry" -> call show_metrics(tab='metrics').
    - User asks to "go back", "back", "back to metrics", or "return to metrics" -> call show_metrics(tab='metrics').
    - User asks to "copy summary" / "copy report" -> call copy_summary.
    - User asks to email/send report -> call send_metrics_email.
    - User asks to "close the summary", "close summary", "close modal", "close metrics", "done", "dismiss", or "hide" -> call close_metrics_modal.
    - User combines commands (e.g. "copy summary and close", "email metrics to user@example.com and close") -> call both tools together.
  * copy_summary: Copy executive session summary to the user's clipboard.
  * close_metrics_modal: Close or dismiss the metrics and summary modal dialog.
  * send_metrics_email: Send metrics email report.
  * reset_tracker: Reset tracker state.

Always be concise, professional, stay strictly in character, and confirm executed actions. If the user asks for multiple actions at once, call all relevant tools together in one response.
"""


def _normalize_coco_name(name: str) -> Optional[str]:
    """Map common aliases or singular/plural to standard COCO class names."""
    n = name.strip().lower()
    # Direct match
    if n in ALL_COCO_CLASSES:
        return n

    # Common aliases & plurals
    aliases = {
        "people": "person",
        "human": "person",
        "humans": "person",
        "cars": "car",
        "phones": "cell phone",
        "cellphone": "cell phone",
        "mobile": "cell phone",
        "mobile phone": "cell phone",
        "mobiles": "cell phone",
        "cups": "cup",
        "glasses": "wine glass",
        "bottles": "bottle",
        "chairs": "chair",
        "dogs": "dog",
        "cats": "cat",
        "bicycles": "bicycle",
        "bikes": "bicycle",
        "motorcycles": "motorcycle",
        "laptops": "laptop",
        "computers": "laptop",
        "screens": "tv",
        "monitors": "tv",
        "tvs": "tv",
        "television": "tv",
        "backpacks": "backpack",
        "bags": "handbag",
        "handbags": "handbag",
        "books": "book",
        "clocks": "clock",
        "watches": "clock",
        "aeroplane": "airplane",
        "aeroplanes": "airplane",
        "planes": "airplane",
        "buses": "bus",
        "trucks": "truck",
        "boats": "boat",
        "traffic lights": "traffic light",
        "potted plants": "potted plant",
        "plants": "potted plant",
        "tables": "dining table",
        "dining tables": "dining table",
    }
    if n in aliases:
        return aliases[n]

    # Plural strip 's'
    if n.endswith("s") and n[:-1] in ALL_COCO_CLASSES:
        return n[:-1]

    # Substring search
    for cls in ALL_COCO_CLASSES:
        if n in cls:
            return cls

    return None


def execute_tool_call(
    func_name: str, args: Dict[str, Any], current_state: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[ChatAction], str]:
    """Translate a function call into a structured ChatAction and summary message."""
    state = current_state or {}

    if func_name == "start_camera":
        return ChatAction(type="START_CAMERA", payload={}, summary="Camera Started"), "Starting the webcam."

    elif func_name == "stop_camera":
        return ChatAction(type="STOP_CAMERA", payload={}, summary="Camera Stopped"), "Stopping the webcam."

    elif func_name == "toggle_detection":
        act = args.get("action", "toggle").lower()
        return (
            ChatAction(type="TOGGLE_DETECTION", payload={"action": act}, summary=f"Detection {act.capitalize()}ed"),
            f"Object detection inference {act}ed.",
        )

    elif func_name == "set_confidence":
        val = float(args.get("threshold", 0.35))
        clamped = max(0.01, min(1.0, val))
        return (
            ChatAction(
                type="SET_CONFIDENCE",
                payload={"value": clamped},
                summary=f"Confidence: {round(clamped * 100)}%",
            ),
            f"Confidence threshold set to {round(clamped * 100)}% ({clamped:.2f}).",
        )

    elif func_name == "set_frame_cadence":
        val = int(args.get("interval_ms", 250))
        # Snap to supported UI options [200, 250, 300, 500]
        options = [200, 250, 300, 500]
        closest = min(options, key=lambda x: abs(x - val))
        return (
            ChatAction(
                type="SET_FRAME_CADENCE",
                payload={"interval_ms": closest},
                summary=f"Cadence: {closest}ms",
            ),
            f"Frame capture cadence set to {closest} ms.",
        )

    elif func_name == "set_min_move_distance":
        val = int(args.get("distance_px", 50))
        clamped = max(25, min(150, val))
        return (
            ChatAction(
                type="SET_MIN_MOVE_DISTANCE",
                payload={"distance_px": clamped},
                summary=f"Min Move: {clamped}px",
            ),
            f"Minimum movement distance set to {clamped} px.",
        )

    elif func_name == "set_settle_dwell_time":
        val = int(args.get("dwell_ms", 1500))
        clamped = max(800, min(3000, val))
        return (
            ChatAction(
                type="SET_SETTLE_DWELL_TIME",
                payload={"dwell_ms": clamped},
                summary=f"Dwell: {clamped / 1000:.1f}s",
            ),
            f"Settle dwell time set to {clamped / 1000:.1f} seconds.",
        )

    elif func_name == "set_coco_classes":
        mode = args.get("mode", "replace")
        raw_classes = args.get("classes", [])
        norm_classes = []
        for c in raw_classes:
            resolved = _normalize_coco_name(str(c))
            if resolved:
                norm_classes.append(resolved)

        if mode == "select_all":
            return (
                ChatAction(
                    type="SET_COCO_CLASSES",
                    payload={"mode": "select_all", "classes": ALL_COCO_CLASSES},
                    summary="All Classes Enabled",
                ),
                "Enabled all 80 COCO classes.",
            )
        elif mode == "clear_all":
            return (
                ChatAction(
                    type="SET_COCO_CLASSES",
                    payload={"mode": "clear_all", "classes": []},
                    summary="Classes Cleared",
                ),
                "Disabled all COCO classes.",
            )
        else:
            return (
                ChatAction(
                    type="SET_COCO_CLASSES",
                    payload={"mode": mode, "classes": norm_classes},
                    summary=f"Classes ({len(norm_classes)}): {', '.join(norm_classes[:3])}{'...' if len(norm_classes) > 3 else ''}",
                ),
                f"Updated active detection classes to: {', '.join(norm_classes) if norm_classes else 'none'}.",
            )

    elif func_name == "toggle_custom_item":
        item_ref = str(args.get("item_name_or_id", ""))
        enable = bool(args.get("enable", True))
        return (
            ChatAction(
                type="TOGGLE_CUSTOM_ITEM",
                payload={"item_ref": item_ref, "enable": enable},
                summary=f"{'Enabled' if enable else 'Disabled'} '{item_ref}'",
            ),
            f"{'Enabled' if enable else 'Disabled'} custom item '{item_ref}'.",
        )

    elif func_name == "delete_custom_item":
        item_ref = str(args.get("item_name_or_id", "")).strip()
        existing_names = state.get("customItemNames", [])
        matched = None
        for n in existing_names:
            if n.lower() == item_ref.lower():
                matched = n
                break
        target_name = matched or item_ref
        return (
            ChatAction(
                type="DELETE_CUSTOM_ITEM",
                payload={"item_ref": target_name},
                summary=f"Deleted '{target_name}'",
            ),
            f"Deleted custom item '{target_name}'.",
        )

    elif func_name == "add_images_to_custom_item":
        item_ref = str(args.get("item_name_or_id", "")).strip()
        existing_names = state.get("customItemNames", [])
        matched = None
        for n in existing_names:
            if n.lower() == item_ref.lower():
                matched = n
                break
        target_name = matched or item_ref
        attached_count = int(state.get("attachedCount", 1))
        photos_str = f"{attached_count} reference photo{'s' if attached_count > 1 else ''}"
        return (
            ChatAction(
                type="ADD_CUSTOM_ITEM_IMAGES",
                payload={"item_ref": target_name},
                summary=f"Added Photos -> '{target_name}'",
            ),
            f"Adding {photos_str} to existing custom item **'{target_name}'**.",
        )

    elif func_name == "set_similarity_threshold":
        val = float(args.get("threshold", 0.65))
        clamped = max(0.1, min(1.0, val))
        return (
            ChatAction(
                type="SET_SIMILARITY_THRESHOLD",
                payload={"value": clamped},
                summary=f"Strictness: {round(clamped * 100)}%",
            ),
            f"Match strictness set to {round(clamped * 100)}% ({clamped:.2f}).",
        )

    elif func_name == "capture_snapshot_as_custom_item":
        raw_name = str(args.get("item_name", "")).strip()
        cleaned_name = " ".join(raw_name.split())
        if not cleaned_name:
            return None, "Please provide a name for your custom item (e.g. 'take snapshot and save as Coffee Mug')."
        if len(cleaned_name) > 60:
            return None, "Item name must be 60 characters or fewer."

        has_attached = bool(state.get("hasAttachedImage", False))
        cam_status = state.get("cameraStatus")
        if cam_status != "active" and not has_attached:
            return (
                None,
                "The camera is currently stopped. Please start the camera first so I can capture a snapshot of your item.",
            )

        existing_names = [str(n).strip().lower() for n in state.get("customItemNames", [])]
        if cleaned_name.lower() in existing_names:
            return (
                None,
                f"A custom item named '{cleaned_name}' already exists. Please choose a unique name or delete the existing item first.",
            )

        attached_count = int(state.get("attachedCount", 1))
        photos_str = (
            f"{attached_count} reference photos"
            if attached_count > 1
            else "1 reference photo"
        )
        action_summary = (
            f"Custom Item ({attached_count}) -> '{cleaned_name}'"
            if has_attached and attached_count > 1
            else f"Custom Item -> '{cleaned_name}'"
            if has_attached
            else f"Snapshot -> '{cleaned_name}'"
        )
        action_msg = (
            f"Registering custom item '{cleaned_name}' from your {photos_str}."
            if has_attached
            else f"Capturing live frame snapshot for '{cleaned_name}'. Please adjust the crop box on screen and confirm to save."
        )
        return (
            ChatAction(
                type="CAPTURE_CUSTOM_ITEM",
                payload={"name": cleaned_name},
                summary=action_summary,
            ),
            action_msg,
        )

    elif func_name == "guard_scene":
        arm = bool(args.get("arm", True))
        return (
            ChatAction(
                type="GUARD_SCENE",
                payload={"arm": arm},
                summary=f"Scene {'Armed' if arm else 'Disarmed'}",
            ),
            f"Scene guarded baseline {'armed' if arm else 'disarmed'}.",
        )

    elif func_name == "dismiss_missing_alert":
        label = str(args.get("item_label", ""))
        return (
            ChatAction(
                type="DISMISS_MISSING_ALERT",
                payload={"label": label},
                summary=f"Dismissed '{label}' alert",
            ),
            f"Dismissed missing alert for '{label}'.",
        )

    elif func_name == "show_metrics":
        raw_tab = str(args.get("tab") or args.get("view") or "metrics").lower().strip()
        is_summary = raw_tab == "summary"
        tab = "summary" if is_summary else "metrics"
        summary_title = "Summary Opened" if is_summary else "Metrics Opened"
        reply_msg = "Opening executive session summary." if is_summary else "Opening session metrics modal."
        return ChatAction(type="SHOW_METRICS", payload={"tab": tab}, summary=summary_title), reply_msg

    elif func_name == "copy_summary":
        return (
            ChatAction(type="COPY_SUMMARY", payload={}, summary="Summary Copied"),
            "Copied executive session summary to your clipboard.",
        )

    elif func_name == "close_metrics_modal":
        return (
            ChatAction(type="CLOSE_METRICS_MODAL", payload={}, summary="Modal Closed"),
            "Closed session metrics and summary modal.",
        )

    elif func_name == "send_metrics_email":
        recip = args.get("recipient_email")
        return (
            ChatAction(
                type="SEND_METRICS_EMAIL",
                payload={"recipient_email": recip},
                summary=f"Emailing Metrics{' to ' + recip if recip else ''}",
            ),
            f"Dispatched session metrics report via email{' to ' + recip if recip else ''}.",
        )

    elif func_name == "reset_tracker":
        return ChatAction(type="RESET_TRACKER", payload={}, summary="Tracker Reset"), "Resetting tracker and scene history."

    return None, ""


def local_intent_fallback(
    message: str,
    current_state: Optional[Dict[str, Any]] = None,
    uploaded_file_name: Optional[str] = None,
    history: Optional[List[ChatMessage]] = None,
) -> ChatResponse:
    """Intelligent semantic regex & keyword intent parser when external LLM is offline or credits depleted."""
    text = message.lower().strip()
    actions: List[ChatAction] = []
    summaries: List[str] = []
    response_lines: List[str] = []

    # Check for affirmative confirmation to previous assistant proposal
    last_assistant_msg = ""
    if history:
        for msg in reversed(history):
            if msg.role == "assistant" and msg.content:
                last_assistant_msg = msg.content.lower()
                break

    is_affirmative = bool(
        re.match(r"^\s*(yes|yep|yeah|sure|ok|okay|please|do it|confirm|proceed|yes please|go ahead)\b", text)
    )
    if is_affirmative and last_assistant_msg:
        if "stop the camera" in last_assistant_msg and (
            "send" in last_assistant_msg or "report" in last_assistant_msg or "email" in last_assistant_msg
        ):
            action_stop, _ = execute_tool_call("stop_camera", {}, current_state)
            action_email, _ = execute_tool_call("send_metrics_email", {}, current_state)
            actions = [a for a in [action_stop, action_email] if a]
            summaries = [a.summary for a in actions if a and a.summary]
            return ChatResponse(
                reply="Stopped the camera and dispatched session metrics report via email.",
                actions=actions,
                action_summaries=summaries,
                used_model="local_intent_parser",
            )
        elif "stop the camera" in last_assistant_msg and "copy" in last_assistant_msg:
            action_stop, _ = execute_tool_call("stop_camera", {}, current_state)
            action_copy, _ = execute_tool_call("copy_summary", {}, current_state)
            actions = [a for a in [action_stop, action_copy] if a]
            summaries = [a.summary for a in actions if a and a.summary]
            return ChatResponse(
                reply="Stopped the camera and copied executive session summary to your clipboard.",
                actions=actions,
                action_summaries=summaries,
                used_model="local_intent_parser",
            )
        elif "stop the camera" in last_assistant_msg and (
            "show" in last_assistant_msg or "view" in last_assistant_msg or "summary" in last_assistant_msg or "metrics" in last_assistant_msg
        ):
            is_summary = "summary" in last_assistant_msg
            target_tab = "summary" if is_summary else "metrics"
            action_stop, _ = execute_tool_call("stop_camera", {}, current_state)
            action_show, _ = execute_tool_call("show_metrics", {"tab": target_tab}, current_state)
            actions = [a for a in [action_stop, action_show] if a]
            summaries = [a.summary for a in actions if a and a.summary]
            reply_text = (
                "Stopped the camera and opened executive session summary."
                if is_summary
                else "Stopped the camera and opened session metrics."
            )
            return ChatResponse(
                reply=reply_text,
                actions=actions,
                action_summaries=summaries,
                used_model="local_intent_parser",
            )

    # 1. Camera Start / Stop
    if re.search(r"\b(start|turn on|open|launch|enable|activate)\b.*\b(camera|webcam|video)\b", text):
        action, msg = execute_tool_call("start_camera", {}, current_state)
        if action:
            actions.append(action)
            summaries.append(action.summary or "")
            response_lines.append("I've started the webcam video stream.")
    elif re.search(r"\b(stop|turn off|close|shut down|disable|deactivate)\b.*\b(camera|webcam|video)\b", text):
        if not re.search(r"\b(metrics|summary|report|stats)\b", text):
            action, msg = execute_tool_call("stop_camera", {}, current_state)
            if action:
                actions.append(action)
                summaries.append(action.summary or "")
                response_lines.append("I've stopped the webcam and recorded your session metrics.")

    # 2. Detection Pause / Resume
    if re.search(r"\b(pause|freeze|suspend)\b.*\b(detection|detecting|yolo)\b", text):
        action, msg = execute_tool_call("toggle_detection", {"action": "pause"}, current_state)
        if action:
            actions.append(action)
            summaries.append(action.summary or "")
            response_lines.append("Detection inference paused.")
    elif re.search(r"\b(resume|unpause|continue)\b.*\b(detection|detecting|yolo)\b", text):
        action, msg = execute_tool_call("toggle_detection", {"action": "resume"}, current_state)
        if action:
            actions.append(action)
            summaries.append(action.summary or "")
            response_lines.append("Detection inference resumed.")

    # 3. Custom Items: Match Strictness / Similarity Threshold (Evaluated BEFORE confidence threshold)
    strictness_match = re.search(
        r"\b(?:match strictness|strictness|matching strictness|match threshold|similarity|similarity threshold|sim threshold|custom item threshold)\b.*?(\d{1,3}(?:\.\d+)?)\s*(%?)",
        text,
    )
    if strictness_match:
        val = float(strictness_match.group(1))
        is_pct = strictness_match.group(2) == "%" or val > 1.0
        parsed_sim = (val / 100.0) if is_pct else val
        action, msg = execute_tool_call(
            "set_similarity_threshold", {"threshold": parsed_sim}, current_state
        )
        if action:
            actions.append(action)
            summaries.append(action.summary or "")
            response_lines.append(msg)
    else:
        # Confidence Threshold (COCO / YOLO)
        conf_match = re.search(
            r"\b(?:confidence|threshold|conf)\b.*?(\d{1,3}(?:\.\d+)?)\s*(%?)", text
        )
        if conf_match and not re.search(r"\b(strict|strictness|similarity|sim)\b", text):
            val = float(conf_match.group(1))
            is_pct = conf_match.group(2) == "%" or val > 1.0
            parsed_conf = (val / 100.0) if is_pct else val
            action, msg = execute_tool_call("set_confidence", {"threshold": parsed_conf}, current_state)
            if action:
                actions.append(action)
                summaries.append(action.summary or "")
                response_lines.append(msg)

    # 4. Cadence / Frame Rate
    cadence_match = re.search(r"\b(?:cadence|interval|fps|frame rate)\b.*?(\d+)\s*(ms|fps)?", text)
    if cadence_match:
        val = int(cadence_match.group(1))
        unit = (cadence_match.group(2) or "").lower()
        if unit == "fps" or (val <= 10 and unit != "ms"):
            # e.g. 5 fps -> 200ms, 4 fps -> 250ms, 3 fps -> 300ms, 2 fps -> 500ms
            if val >= 5:
                interval_ms = 200
            elif val == 4:
                interval_ms = 250
            elif val == 3:
                interval_ms = 300
            else:
                interval_ms = 500
        else:
            interval_ms = val
        action, msg = execute_tool_call("set_frame_cadence", {"interval_ms": interval_ms}, current_state)
        if action:
            actions.append(action)
            summaries.append(action.summary or "")
            response_lines.append(msg)

    # 5. Min Move Distance
    dist_match = re.search(r"\b(?:move distance|min move|distance threshold)\b.*?(\d+)", text)
    if dist_match:
        dist_px = int(dist_match.group(1))
        action, msg = execute_tool_call("set_min_move_distance", {"distance_px": dist_px}, current_state)
        if action:
            actions.append(action)
            summaries.append(action.summary or "")
            response_lines.append(msg)

    # 6. Settle Dwell Time
    dwell_match = re.search(r"\b(?:dwell|settle time|dwell time)\b.*?(\d+(?:\.\d+)?)\s*(s|sec|seconds|ms)?", text)
    if dwell_match:
        raw_val = float(dwell_match.group(1))
        unit = (dwell_match.group(2) or "").lower()
        dwell_ms = int(raw_val * 1000) if ("s" in unit or raw_val <= 10) else int(raw_val)
        action, msg = execute_tool_call("set_settle_dwell_time", {"dwell_ms": dwell_ms}, current_state)
        if action:
            actions.append(action)
            summaries.append(action.summary or "")
            response_lines.append(msg)

    # 7. Guard Scene / Disarm
    if re.search(r"\b(guard|arm|lock)\b.*\b(scene|baseline|room)\b", text):
        action, msg = execute_tool_call("guard_scene", {"arm": True}, current_state)
        if action:
            actions.append(action)
            summaries.append(action.summary or "")
            response_lines.append("Guarded the scene! Current detected objects locked as baseline.")
    elif re.search(r"\b(disarm|unguard|unlock|un-guard)\b.*\b(scene|baseline|room)\b", text):
        action, msg = execute_tool_call("guard_scene", {"arm": False}, current_state)
        if action:
            actions.append(action)
            summaries.append(action.summary or "")
            response_lines.append("Disarmed the scene.")

    # 8. Custom Items: Delete / Remove (Evaluated first to cleanly catch "delete My Item" or "remove Coffee Mug")
    is_delete_req = bool(re.search(r"\b(?:delete|remove)\b", text))
    existing_names = current_state.get("customItemNames", [])
    has_attached = bool(current_state.get("hasAttachedImage", False))
    is_custom_item_add = False

    is_add_to_existing = bool(
        re.search(r"\b(?:add|append)\b.*?\b(?:to|into|for)\b", text)
        or re.search(r"\b(?:add|append)\b.*?\b(?:more|another|extra)\s+(?:photos?|images?|snapshots?|references?)\b", text)
    )

    if is_delete_req and not re.search(r"\b(all classes|coco|classes|events?|logs?)\b", text):
        matched_del_name = None
        for ename in existing_names:
            if re.search(rf"\b(?:delete|remove)\b.*?\b{re.escape(ename.lower())}\b", text):
                matched_del_name = ename
                break
        if not matched_del_name:
            quoted_del = re.search(r"['\"]([^'\"]+)['\"]", message)
            if quoted_del:
                matched_del_name = quoted_del.group(1).strip()
            else:
                m_del = re.search(
                    r"\b(?:delete|remove)\b\s+(?:custom\s+item\s+|item\s+)?([a-zA-Z0-9 _-]+)$",
                    message.strip(),
                    re.IGNORECASE,
                )
                if m_del:
                    matched_del_name = m_del.group(1).strip()

        if matched_del_name:
            action, msg = execute_tool_call(
                "delete_custom_item", {"item_name_or_id": matched_del_name}, current_state
            )
            if action:
                actions.append(action)
                summaries.append(action.summary or "")
                response_lines.append(msg)
            elif msg:
                response_lines.append(msg)

    # 9. Custom Items: Add photos to existing item
    elif is_add_to_existing:
        matched_target = None
        for ename in existing_names:
            if re.search(rf"\b{re.escape(ename.lower())}\b", text):
                matched_target = ename
                break
        if not matched_target:
            quoted_target = re.search(r"['\"]([^'\"]+)['\"]", message)
            if quoted_target:
                matched_target = quoted_target.group(1).strip()
            else:
                m_target = re.search(
                    r"\b(?:to|into|for)\s+(?:custom\s+item\s+|item\s+)?['\"]?([a-zA-Z0-9 _-]+?)['\"]?$",
                    message.strip(),
                    re.IGNORECASE,
                )
                if m_target:
                    matched_target = m_target.group(1).strip()

        if matched_target:
            action, msg = execute_tool_call(
                "add_images_to_custom_item",
                {"item_name_or_id": matched_target},
                current_state,
            )
            if action:
                actions.append(action)
                summaries.append(action.summary or "")
                response_lines.append(msg)
            elif msg:
                response_lines.append(msg)
        else:
            response_lines.append("Please specify which existing custom item you would like to add photos to.")

    # 10. Custom Items: Create new item / Snapshot
    elif (
        re.search(
            r"\b(capture|take snapshot|snapshot|save frame|add as item|add custom item|register item|save as custom item)\b",
            text,
        )
        or re.search(
            r"\b(save|register|add)\b.*\b(custom item|item|reference photo|reference photos|photo|photos)\b",
            text,
        )
        or (
            has_attached
            and len(actions) == 0
            and len(message.strip()) <= 60
            and not re.search(
                r"\b(start|stop|confidence|cadence|fps|dwell|guard|disarm|metrics|summary|delete|remove|strict|strictness|similarity|set|add|clear)\b",
                text,
            )
        )
    ):
        is_custom_item_add = True
        item_name = ""
        # Extract quoted item name from the original message to preserve user casing (e.g. 'My Item')
        quoted_orig = re.search(r"['\"]([^'\"]+)['\"]", message)
        if quoted_orig:
            raw_quoted = quoted_orig.group(1).strip()
            if raw_quoted:
                item_name = raw_quoted
        else:
            m = re.search(
                r"\b(?:add\s+custom\s+item|save\s+as(?:\s+custom\s+item)?|named|called|as(?:\s+my)?(?:\s+custom)?\s+item|as)\s+([a-zA-Z0-9 _-]+)",
                message,
                re.IGNORECASE,
            )
            if m:
                raw_name = m.group(1).strip()
                stripped = re.sub(r"^(custom\s+item|my\s+item|item)\s+", "", raw_name, flags=re.IGNORECASE).strip()
                item_name = stripped.title() if stripped else raw_name.title()
            elif has_attached:
                cand_name = message.strip()
                if 0 < len(cand_name) <= 60:
                    item_name = cand_name.title()

        if item_name:
            action, msg = execute_tool_call(
                "capture_snapshot_as_custom_item", {"item_name": item_name}, current_state
            )
            if action:
                actions.append(action)
                summaries.append(action.summary or "")
                response_lines.append(msg)
            elif msg:
                response_lines.append(msg)
        else:
            response_lines.append(
                "Please provide a name for your custom item (e.g. 'take snapshot and save as Coffee Mug')."
            )

    # 11. COCO Classes selection (only if not adding custom item)
    elif re.search(r"\b(clear all|clear classes|disable all|no classes|remove all|reset classes)\b", text):
        action, msg = execute_tool_call("set_coco_classes", {"mode": "clear_all"}, current_state)
        if action:
            actions.append(action)
            summaries.append(action.summary or "")
            response_lines.append("Cleared all COCO classes.")
    elif re.search(r"\b(all classes|select all|enable all|every class)\b", text):
        action, msg = execute_tool_call("set_coco_classes", {"mode": "select_all"}, current_state)
        if action:
            actions.append(action)
            summaries.append(action.summary or "")
            response_lines.append("Enabled all 80 COCO classes.")
    elif not is_custom_item_add and not is_add_to_existing:
        # Check for specific classes mentioned
        detected_classes: List[str] = []
        for word in ALL_COCO_CLASSES:
            # Match whole words only
            pattern = rf"\b{re.escape(word)}s?\b"
            if re.search(pattern, text):
                detected_classes.append(word)

        # Check aliases
        alias_pairs = [
            ("person", r"\b(people|humans?)\b"),
            ("cell phone", r"\b(phones?|cellphones?|mobiles?)\b"),
            ("tv", r"\b(screens?|monitors?|television)\b"),
            ("bicycle", r"\b(bikes?)\b"),
            ("dining table", r"\b(tables?)\b"),
            ("laptop", r"\b(computers?)\b"),
            ("potted plant", r"\b(plants?)\b"),
        ]
        for canonical, alias_pat in alias_pairs:
            if re.search(alias_pat, text) and canonical not in detected_classes:
                detected_classes.append(canonical)

        if detected_classes:
            is_remove = bool(re.search(r"\b(remove|disable|stop detecting|don't detect|do not detect)\b", text))
            is_add = bool(re.search(r"\b(add|also detect|include)\b", text))
            mode = "remove" if is_remove else ("add" if is_add else "replace")
            action, msg = execute_tool_call(
                "set_coco_classes", {"mode": mode, "classes": detected_classes}, current_state
            )
            if action:
                actions.append(action)
                summaries.append(action.summary or "")
                response_lines.append(
                    f"{'Removed' if mode == 'remove' else ('Added' if mode == 'add' else 'Set active classes to')}: {', '.join(detected_classes)}."
                )


    # 12. Metrics & Reporting
    state = current_state or {}
    cam_active = state.get("cameraStatus") == "active"
    has_metrics = state.get("hasMetrics", False)

    # Navigation back to metrics grid
    if re.search(r"\b(go back|back to metrics|return to metrics|switch to metrics)\b", text):
        action, msg = execute_tool_call("show_metrics", {"tab": "metrics"}, current_state)
        if action:
            actions.append(action)
            summaries.append(action.summary or "")
            response_lines.append("Switched view back to session metrics.")

    # Close modal / summary / metrics dialog
    if (
        re.search(r"\b(close|dismiss|hide|exit)\b.*\b(summary|metrics|modal|report|dialog|window)\b", text)
        or text in ["close", "close modal", "close summary", "done", "dismiss", "hide"]
    ):
        action, msg = execute_tool_call("close_metrics_modal", {}, current_state)
        if action:
            actions.append(action)
            summaries.append(action.summary or "")
            response_lines.append("Closed session metrics and summary modal.")
        elif msg:
            response_lines.append(msg)

    # User explicitly asking to stop camera and send/email metrics
    if re.search(r"\b(stop|turn off|close)\b.*\b(camera|webcam)\b", text) and (
        re.search(r"\b(email|send)\b.*\b(metrics|summary|report|stats)\b", text)
        or re.search(r"\b(?:email|send report|send metrics)\b", text)
    ):
        action_stop, msg_stop = execute_tool_call("stop_camera", {}, current_state)
        if action_stop:
            actions.append(action_stop)
            summaries.append(action_stop.summary or "")
        email_match = re.search(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", text)
        target_email = email_match.group(1) if email_match else None
        action_email, msg_email = execute_tool_call(
            "send_metrics_email", {"recipient_email": target_email} if target_email else {}, current_state
        )
        if action_email:
            actions.append(action_email)
            summaries.append(action_email.summary or "")
        response_lines.append(
            f"Stopped the camera and dispatched session metrics report via email{' to ' + target_email if target_email else ''}."
        )

    elif re.search(r"\b(show|open|view|display)\b.*\b(metrics|summary|report|stats)\b", text) or re.search(r"\b(executive summary|session summary)\b", text):
        is_stop_and_show = bool(re.search(r"\b(stop|turn off|close)\b.*\b(camera|webcam)\b", text))
        wants_summary = bool(re.search(r"\b(summary|executive|report)\b", text))
        target_tab = "summary" if wants_summary else "metrics"
        tab_label = "executive session summary" if wants_summary else "session metrics"

        if is_stop_and_show:
            action_stop, msg_stop = execute_tool_call("stop_camera", {}, current_state)
            if action_stop:
                actions.append(action_stop)
                summaries.append(action_stop.summary or "")
            action_show, msg_show = execute_tool_call("show_metrics", {"tab": target_tab}, current_state)
            if action_show:
                actions.append(action_show)
                summaries.append(action_show.summary or "")
            response_lines.append(f"Stopped the camera and opened {tab_label}.")
        elif cam_active:
            response_lines.append(
                f"Session metrics and the executive summary will be available after stopping the camera. "
                f"Would you like me to stop the camera and show your {tab_label} now?"
            )
        elif not has_metrics:
            response_lines.append(
                f"No session metrics recorded yet. Session metrics and the executive summary will be available after starting and stopping the camera."
            )
        else:
            action, msg = execute_tool_call("show_metrics", {"tab": target_tab}, current_state)
            if action:
                actions.append(action)
                summaries.append(action.summary or "")
                response_lines.append(f"Opened {tab_label}.")

    elif re.search(r"\b(copy|clipboard)\b.*\b(summary|report|executive)\b", text):
        is_stop_and_copy = bool(re.search(r"\b(stop|turn off|close)\b.*\b(camera|webcam)\b", text))
        if is_stop_and_copy:
            action_stop, _ = execute_tool_call("stop_camera", {}, current_state)
            if action_stop:
                actions.append(action_stop)
                summaries.append(action_stop.summary or "")
            action_copy, _ = execute_tool_call("copy_summary", {}, current_state)
            if action_copy:
                actions.append(action_copy)
                summaries.append(action_copy.summary or "")
            response_lines.append("Stopped the camera and copied executive session summary to your clipboard.")
        elif cam_active:
            response_lines.append(
                "Session metrics and the executive summary will be available after stopping the camera. "
                "Would you like me to stop the camera and copy your executive session summary now?"
            )
        elif not has_metrics:
            response_lines.append(
                "No session metrics recorded yet. Start and stop the camera first to generate a session summary."
            )
        else:
            action, msg = execute_tool_call("copy_summary", {}, current_state)
            if action:
                actions.append(action)
                summaries.append(action.summary or "")
                response_lines.append("Copied executive session summary to your clipboard.")

    elif re.search(r"\b(?:email|send report|send metrics)\b", text) or (
        re.search(r"\b(email|send)\b", text) and re.search(r"\b(metrics|summary|report|stats)\b", text)
    ):
        if cam_active:
            response_lines.append(
                "I cannot send a metrics report yet because the camera is currently active and no session data has been finalized. "
                "Session metrics are generated only after you stop the camera.\n\n"
                "Would you like me to stop the camera now so we can finalize the session and send the report?"
            )
        elif not has_metrics:
            response_lines.append(
                "No session metrics recorded yet. Session metrics will be available after starting and stopping the camera."
            )
        else:
            email_match = re.search(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", text)
            target_email = email_match.group(1) if email_match else None
            action, msg = execute_tool_call(
                "send_metrics_email",
                {"recipient_email": target_email} if target_email else {},
                current_state,
            )
            if action:
                actions.append(action)
                summaries.append(action.summary or "")
                response_lines.append(msg)

    # 13. Reset Tracker
    if re.search(r"\b(reset|clear)\b.*\b(tracker|tracking|relocations|movement|scene)\b", text):
        action, msg = execute_tool_call("reset_tracker", {}, current_state)
        if action:
            actions.append(action)
            summaries.append(action.summary or "")
            response_lines.append("Tracker state and movement events successfully reset.")

    # If specific response lines were generated, use them
    if response_lines:
        reply = "\n".join(response_lines)
    elif not actions:
        if re.search(r"\b(use\s*effect|react|hook|javascript|python|coding|closure|promise|html|css|redux|who is|who was|tell me a joke|write a poem|weather)\b", text):
            reply = (
                "I'm your **Vision AI Copilot**, dedicated specifically to controlling and monitoring this webcam object detection dashboard.\n\n"
                "I focus on dashboard operations rather than general programming or trivia. I can help you with:\n"
                "* 📹 **Camera:** *\"Start camera\"* or *\"Stop camera\"*\n"
                "* 🎯 **Classes:** *\"Only detect person and cup\"* or *\"Select all classes\"*\n"
                "* 📦 **Custom Items:** *\"Take snapshot and save as 'My Coffee Mug'\"*\n"
                "* 🎚️ **Sliders:** *\"Set confidence to 60%\"* or *\"Change frame rate to 5 FPS\"*\n"
                "* 🛡️ **Guarding:** *\"Guard scene\"* or *\"Disarm scene\"*\n"
                "* 📊 **Metrics:** *\"Show metrics\"* or *\"Email report to user@example.com\"*"
            )
        else:
            reply = (
                "I'm your **Vision AI Copilot**. I can control everything on your dashboard. Try asking me:\n\n"
                "* 📹 **Camera:** *\"Start camera\"* or *\"Stop camera\"*\n"
                "* 🎯 **Classes:** *\"Only detect person and cup\"*, *\"Add chair\"*, or *\"Select all classes\"*\n"
                "* 📦 **Custom Items:** *\"Take snapshot and save as 'My Coffee Mug'\"*\n"
                "* 🎚️ **Sliders:** *\"Set confidence to 60%\"* or *\"Change frame rate to 5 FPS\"*\n"
                "* 🛡️ **Guarding:** *\"Guard scene\"* or *\"Disarm scene\"*\n"
                "* 📊 **Metrics:** *\"Show metrics\"* or *\"Email report to user@example.com\"*"
            )
    else:
        reply = "\n".join(response_lines)

    return ChatResponse(
        reply=reply,
        actions=actions,
        action_summaries=summaries,
        used_model="local_intent_parser",
    )


async def call_gemini_api(
    message: str,
    history: List[ChatMessage],
    current_state: Optional[Dict[str, Any]],
    settings: Settings,
    image_bytes: Optional[bytes] = None,
    image_mime_type: Optional[str] = None,
) -> ChatResponse:
    """Send user request to Google Gemini with function declarations for tool calling."""
    if not settings.gemini_api_key or not settings.gemini_api_key.strip():
        logger.info("No GEMINI_API_KEY set; using local intent parser.")
        return local_intent_fallback(message, current_state, history=history)

    api_key = (settings.gemini_api_key or "").strip()
    model = (settings.gemini_model or "gemini-3.1-flash-lite").strip()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    # Build contents history
    contents: List[Dict[str, Any]] = []
    # Include up to last 6 history messages
    for msg in history[-6:]:
        gemini_role = "user" if msg.role == "user" else "model"
        contents.append({"role": gemini_role, "parts": [{"text": msg.content}]})

    # Prepare current user turn
    user_parts: List[Dict[str, Any]] = []

    # If an image was uploaded with the prompt
    if image_bytes and image_mime_type:
        b64_data = base64.b64encode(image_bytes).decode("utf-8")
        user_parts.append({
            "inlineData": {
                "mimeType": image_mime_type,
                "data": b64_data,
            }
        })

    # Add context about current UI state if available
    state_desc = ""
    if current_state:
        existing_items = current_state.get("customItemNames", [])
        state_desc = (
            f"\n[Current System State: Camera={current_state.get('cameraStatus', 'unknown')}, "
            f"Detecting={current_state.get('isDetecting')}, Confidence={current_state.get('confidence')}, "
            f"ActiveCOCO={current_state.get('selectedClasses', [])}, CustomItems={existing_items}, "
            f"Guarded={current_state.get('isGuarded')}, hasMetrics={current_state.get('hasMetrics', False)}, "
            f"AttachedImages={current_state.get('hasAttachedImage', False)} ({current_state.get('attachedCount', 0)} files), "
            f"ModalOpen={current_state.get('isMetricsModalOpen', False)}, ModalTab={current_state.get('metricsModalTab', 'metrics')}]"
        )

    user_parts.append({"text": message + state_desc})
    contents.append({"role": "user", "parts": user_parts})

    payload = {
        "contents": contents,
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "tools": [{"functionDeclarations": GEMINI_TOOLS_DECLARATION}],
        "toolConfig": {"functionCallingConfig": {"mode": "AUTO"}},
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 600,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            resp = await client.post(url, headers=headers, json=payload)

        if resp.status_code == 200:
            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return local_intent_fallback(message, current_state, history=history)

            first_candidate = candidates[0]
            content = first_candidate.get("content", {})
            parts = content.get("parts", [])

            actions: List[ChatAction] = []
            summaries: List[str] = []
            text_parts: List[str] = []

            function_calls = [p["functionCall"] for p in parts if "functionCall" in p]
            fn_names = {fc.get("name") for fc in function_calls}

            cam_active = (current_state or {}).get("cameraStatus") == "active"
            has_metrics = (current_state or {}).get("hasMetrics", False)

            msg_lower = message.lower().strip()
            is_explicit_stop = bool(re.search(r"\b(stop|turn off|close)\b.*\b(camera|webcam)\b", msg_lower)) or msg_lower in ["yes", "y", "sure", "ok", "please do"]

            for part in parts:
                if "text" in part and part["text"].strip():
                    text_parts.append(part["text"].strip())
                elif "functionCall" in part:
                    fc = part["functionCall"]
                    fn_name = fc.get("name", "")
                    fn_args = fc.get("args", {})

                    # If camera is active and user did not explicitly ask/confirm to stop camera, disallow preemptive stop_camera, show_metrics, or send_metrics_email
                    if cam_active and not has_metrics and not is_explicit_stop:
                        if fn_name in ("show_metrics", "stop_camera"):
                            if not any("Session metrics and the executive summary" in t for t in text_parts):
                                text_parts.append(
                                    "Session metrics and the executive summary will be available after stopping the camera. "
                                    "Would you like me to stop the camera and show your executive session summary now?"
                                )
                            continue
                        elif fn_name == "send_metrics_email":
                            if not any("I cannot send a metrics report" in t for t in text_parts):
                                text_parts.append(
                                    "I cannot send a metrics report yet because the camera is currently active and no session data has been finalized. "
                                    "Session metrics are generated only after you stop the camera.\n\n"
                                    "Would you like me to stop the camera now so we can finalize the session and send the report?"
                                )
                            continue

                    action, summary_msg = execute_tool_call(fn_name, fn_args, current_state)
                    if action:
                        actions.append(action)
                        if action.summary:
                            summaries.append(action.summary)
                    elif summary_msg:
                        text_parts.append(summary_msg)

            final_reply = "\n\n".join(text_parts) if text_parts else ""
            if not final_reply:
                if actions:
                    final_reply = "Executed your request: " + ", ".join(summaries) + "."
                else:
                    final_reply = "Understood."

            return ChatResponse(
                reply=final_reply,
                actions=actions,
                action_summaries=summaries,
                used_model=model,
            )

        elif resp.status_code == 402:
            logger.warning(
                "Gemini API returned 402 (prepayment credits depleted); falling back to local intent parser."
            )
            fallback = local_intent_fallback(message, current_state, history=history)
            fallback.reply += "\n\n> ℹ️ *Note: Gemini reported that prepayment credits on this key's project are depleted. I executed your commands instantly using the built-in semantic parser. Switch your project in Google AI Studio to the Free Tier to re-enable cloud LLM output!*"
            return fallback

        else:
            logger.warning(
                "Gemini API returned status %d (%s); using local intent parser.",
                resp.status_code,
                resp.text[:200],
            )
            return local_intent_fallback(message, current_state, history=history)

    except Exception as exc:
        logger.exception("Error calling Gemini API: %s; falling back to local intent parser.", exc)
        return local_intent_fallback(message, current_state, history=history)
