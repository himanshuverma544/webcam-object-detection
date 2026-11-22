from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A single turn in the conversation."""

    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message text")
    timestamp: Optional[str] = Field(default=None, description="ISO-8601 or clock timestamp")


class ChatAction(BaseModel):
    """An executable UI or backend action returned by the chatbot."""

    type: str = Field(
        ...,
        description=(
            "Action identifier: START_CAMERA, STOP_CAMERA, TOGGLE_DETECTION, "
            "SET_CONFIDENCE, SET_FRAME_CADENCE, SET_MIN_MOVE_DISTANCE, "
            "SET_SETTLE_DWELL_TIME, SET_COCO_CLASSES, TOGGLE_CUSTOM_ITEM, "
            "DELETE_CUSTOM_ITEM, SET_SIMILARITY_THRESHOLD, CAPTURE_CUSTOM_ITEM, "
            "CREATE_CUSTOM_ITEM, GUARD_SCENE, DISMISS_MISSING_ALERT, "
            "SHOW_METRICS, SEND_METRICS_EMAIL, RESET_TRACKER"
        ),
    )
    payload: Dict[str, Any] = Field(default_factory=dict, description="Action arguments")
    summary: Optional[str] = Field(default=None, description="Short summary badge text")


class ChatRequest(BaseModel):
    """Incoming chat request payload."""

    message: str = Field(..., description="User message text")
    history: List[ChatMessage] = Field(
        default_factory=list, description="Prior conversation history"
    )
    current_state: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Current client state (cameraStatus, selectedClasses, confidence, etc.)",
    )


class ChatResponse(BaseModel):
    """Structured response from the AI Copilot."""

    reply: str = Field(..., description="Assistant's conversational markdown reply")
    actions: List[ChatAction] = Field(
        default_factory=list, description="Actions dispatched to the UI/backend"
    )
    action_summaries: List[str] = Field(
        default_factory=list, description="List of short confirmation badges"
    )
    used_model: Optional[str] = Field(
        default=None, description="Name of LLM model or 'local_parser' used"
    )

