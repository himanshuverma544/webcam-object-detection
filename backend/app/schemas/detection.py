from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check and model readiness response."""

    status: str = Field(
        ...,
        description="API health status ('ok' when model is ready, 'degraded' otherwise)",
        examples=["ok"],
    )
    model_loaded: bool = Field(
        ...,
        description="True only after the YOLO model has been loaded successfully",
    )
    model_ready: bool = Field(
        ...,
        description="True only when the model is loaded and ready for inference",
    )
    model_name: str = Field(
        ...,
        description="Configured YOLO model identifier",
        examples=["yolo11n.pt"],
    )
    device: str = Field(
        ...,
        description="Compute device used for inference (cpu, mps, or cuda)",
        examples=["cpu"],
    )
    classes_count: int = Field(
        ...,
        description="Number of object classes supported by the loaded model",
        examples=[80],
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if model initialization failed",
    )


class ClassesResponse(BaseModel):
    """Supported COCO classes response."""

    count: int = Field(..., description="Total number of supported classes", examples=[80])
    classes: Dict[int, str] = Field(
        ...,
        description="Mapping of COCO class ID to class label",
        examples=[{0: "person", 1: "bicycle", 2: "car"}],
    )
    class_names: List[str] = Field(
        ...,
        description="Ordered list of COCO class names",
        examples=[["person", "bicycle", "car"]],
    )


class ImageDimensions(BaseModel):
    """Pixel dimensions of the uploaded image."""

    width: int = Field(..., gt=0, description="Image width in pixels", examples=[640])
    height: int = Field(..., gt=0, description="Image height in pixels", examples=[480])


class DetectionItem(BaseModel):
    """A single detected object in an image frame."""

    label: str = Field(..., description="Detected COCO class name", examples=["person"])
    class_id: int = Field(..., ge=0, description="Detected COCO class ID", examples=[0])
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Detection confidence score between 0.0 and 1.0",
        examples=[0.892],
    )
    bbox: List[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Bounding box coordinates [x1, y1, x2, y2] in pixels relative to the uploaded image",
        examples=[[120.5, 45.0, 340.2, 410.8]],
    )
    track_id: Optional[int] = Field(
        default=None,
        description="Persistent tracking ID assigned by ByteTrack across video frames",
        examples=[1],
    )
    is_custom: bool = Field(
        default=False,
        description="True when matched via a user-uploaded custom reference item",
    )
    custom_item_id: Optional[str] = Field(
        default=None,
        description="Identifier of the matched custom reference item",
    )
    similarity: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Visual instance similarity score between detected crop and reference item",
    )


class ResetTrackResponse(BaseModel):
    """Response returned when resetting tracker state."""

    status: str = Field(..., description="Execution status ('ok')", examples=["ok"])
    message: str = Field(
        ...,
        description="Human-readable status confirmation",
        examples=["Tracker state successfully reset"],
    )


class DetectionResponse(BaseModel):
    """Structured response returned by POST /detect."""

    image_width: int = Field(..., gt=0, description="Uploaded image width in pixels")
    image_height: int = Field(..., gt=0, description="Uploaded image height in pixels")
    image_dimensions: ImageDimensions = Field(
        ..., description="Structured width and height of the uploaded image"
    )
    inference_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Model inference execution latency in milliseconds",
        examples=[42.6],
    )
    confidence_threshold: float = Field(
        ...,
        ge=0.01,
        le=1.0,
        description="Confidence threshold applied during inference",
        examples=[0.35],
    )
    detection_count: int = Field(
        ...,
        ge=0.0,
        description="Total number of objects detected in the frame",
        examples=[2],
    )
    detections: List[DetectionItem] = Field(
        default_factory=list,
        description="List of detected objects sorted by descending confidence",
    )


class ErrorResponse(BaseModel):
    """Standard error response payload."""

    detail: str = Field(..., description="Human-readable error description")


class EmailMetricsRequest(BaseModel):
    """Request payload for sending session metrics report via email."""

    recipient_email: Optional[str] = Field(
        default=None,
        description="Target email address to receive metrics report (falls back to DEFAULT_RECIPIENT_EMAIL if omitted)",
        examples=["user@example.com"],
    )
    metrics: Dict[str, Any] = Field(
        ...,
        description="Complete structured session metrics data payload",
    )


class EmailMetricsResponse(BaseModel):
    """Response payload after dispatching metrics report via email."""

    status: str = Field(
        ...,
        description="Status flag indicating delivery outcome ('sent' or 'ok')",
        examples=["sent"],
    )
    message: str = Field(
        ...,
        description="Human-readable delivery confirmation message",
        examples=["Metrics report email sent successfully"],
    )
    recipient: str = Field(
        ...,
        description="Destination email address where report was dispatched",
        examples=["user@example.com"],
    )


class CustomItemOut(BaseModel):
    """Metadata and thumbnail preview for a registered custom reference item."""

    id: str = Field(..., description="Unique custom item identifier", examples=["item_a1b2c3d4"])
    name: str = Field(..., description="User-assigned item name", examples=["My Office Chair"])
    class_id: int = Field(..., ge=1000, description="Assigned custom class ID (>= 1000)")
    image_count: int = Field(..., ge=1, description="Number of reference photos indexed for this item")
    thumbnail_base64: str = Field(
        ...,
        description="Base64 JPEG data URL preview of the item's primary reference image",
    )
    created_at: str = Field(..., description="ISO-8601 UTC creation timestamp")


class CustomItemsListResponse(BaseModel):
    """List of registered custom reference items."""

    count: int = Field(..., ge=0, description="Number of saved custom reference items")
    items: List[CustomItemOut] = Field(default_factory=list)
    yoloe_ready: bool = Field(
        default=False,
        description="True when YOLOE visual-prompt engine has been initialized",
    )
    device: str = Field(default="cpu", description="Compute device used for custom visual detection")


class CustomItemDeleteResponse(BaseModel):
    """Confirmation payload after deleting a custom reference item."""

    status: str = Field(..., examples=["ok"])
    message: str = Field(..., examples=["Custom item deleted"])
    deleted_id: str = Field(..., examples=["item_a1b2c3d4"])


class AutoBoxResponse(BaseModel):
    """Suggested foreground bounding box returned by POST /custom-items/auto-box."""

    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)
    bbox_norm: List[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Normalized foreground box [x1, y1, x2, y2] in [0.0, 1.0]",
    )
    bbox_px: List[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Pixel foreground box [x1, y1, x2, y2]",
    )
    method: str = Field(default="grabcut", description="Algorithm used to estimate foreground box")
