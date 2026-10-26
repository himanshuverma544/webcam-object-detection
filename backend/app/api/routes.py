from __future__ import annotations

import logging
from typing import Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)

from app.core.config import Settings, get_settings
from app.schemas.detection import (
    ClassesResponse,
    DetectionResponse,
    ErrorResponse,
    HealthResponse,
)
from app.services.detector import YOLOObjectDetector

logger = logging.getLogger(__name__)

router = APIRouter()


def get_detector(request: Request) -> YOLOObjectDetector:
    """Retrieve the application's singleton YOLOObjectDetector instance."""
    detector: Optional[YOLOObjectDetector] = getattr(
        request.app.state, "detector", None
    )
    if detector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Object detection service is not initialized.",
        )
    return detector


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check and model readiness status",
)
async def health_check(
    detector: YOLOObjectDetector = Depends(get_detector),
) -> HealthResponse:
    """Return API status and whether the YOLO11 model has finished loading."""
    is_ready = detector.is_ready
    classes = detector.get_classes()
    return HealthResponse(
        status="ok" if is_ready else "degraded",
        model_loaded=is_ready,
        model_ready=is_ready,
        model_name=detector.model_path,
        device=detector.device,
        classes_count=len(classes),
        error=detector.load_error if not is_ready else None,
    )


@router.get(
    "/classes",
    response_model=ClassesResponse,
    summary="List supported COCO object classes",
)
async def list_classes(
    detector: YOLOObjectDetector = Depends(get_detector),
) -> ClassesResponse:
    """Return the dictionary and ordered list of supported COCO class names."""
    classes = detector.get_classes()
    ordered_names = [classes[k] for k in sorted(classes.keys())]
    return ClassesResponse(
        count=len(classes),
        classes=classes,
        class_names=ordered_names,
    )


@router.post(
    "/detect",
    response_model=DetectionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file type or corrupted image"},
        413: {"model": ErrorResponse, "description": "Image exceeds maximum upload size"},
        422: {"model": ErrorResponse, "description": "Validation error (e.g. invalid confidence)"},
        500: {"model": ErrorResponse, "description": "Inference execution error"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
    summary="Run real-time YOLO11 object detection on an uploaded image frame",
)
async def detect_objects(
    request: Request,
    file: UploadFile = File(..., description="JPEG, PNG, or WebP image frame"),
    confidence: Optional[float] = Query(
        default=None,
        ge=0.01,
        le=1.0,
        description="Confidence threshold between 0.01 and 1.0 (default: 0.35)",
    ),
    confidence_form: Optional[float] = Form(
        default=None,
        ge=0.01,
        le=1.0,
        description="Optional form field for confidence threshold between 0.01 and 1.0",
    ),
    detector: YOLOObjectDetector = Depends(get_detector),
    settings: Settings = Depends(get_settings),
) -> DetectionResponse:
    """Validate uploaded frame, decode with OpenCV, and run YOLO11 inference."""
    # Ensure model is loaded before attempting inference
    if not detector.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Object detection model is not ready. Please check /health.",
        )

    # Resolve confidence threshold from query param, form field, or default setting
    resolved_confidence = (
        confidence
        if confidence is not None
        else (
            confidence_form
            if confidence_form is not None
            else settings.default_confidence
        )
    )

    # Also check raw form field named 'confidence' if sent via multipart form instead of query
    if confidence is None and confidence_form is None:
        try:
            form_data = await request.form()
            raw_conf = form_data.get("confidence")
            if raw_conf is not None and str(raw_conf).strip() != "":
                parsed_conf = float(str(raw_conf))
                if parsed_conf < 0.01 or parsed_conf > 1.0:
                    raise HTTPException(
                        status_code=422,
                        detail="Confidence threshold must be between 0.01 and 1.0.",
                    )
                resolved_confidence = parsed_conf
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail="Confidence threshold must be a valid number between 0.01 and 1.0.",
            ) from exc
        except Exception:
            pass

    if resolved_confidence < 0.01 or resolved_confidence > 1.0:
        raise HTTPException(
            status_code=422,
            detail="Confidence threshold must be between 0.01 and 1.0.",
        )

    # Validate file MIME type
    content_type = (file.content_type or "").lower().strip()
    if content_type not in settings.allowed_content_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file type '{content_type or 'unknown'}'. "
                "Allowed image formats are JPEG, PNG, and WebP."
            ),
        )

    # Read and validate file size
    try:
        contents = await file.read()
    except Exception as exc:
        logger.warning("Failed to read uploaded file stream: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read uploaded image file.",
        ) from exc
    finally:
        await file.close()

    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded image file is empty.",
        )

    if len(contents) > settings.max_upload_size_bytes:
        max_mb = round(settings.max_upload_size_bytes / (1024 * 1024), 2)
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded image exceeds maximum size limit of {max_mb} MB.",
        )

    # Decode image and run YOLO inference
    try:
        return detector.detect(
            image_bytes=contents,
            confidence=resolved_confidence,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc) or "Invalid or corrupted image file.",
        ) from exc
    except RuntimeError as exc:
        logger.error("Inference runtime error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Object detection inference failed on the server.",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error during object detection.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during object detection.",
        ) from exc
