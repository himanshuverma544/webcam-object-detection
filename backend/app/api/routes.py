from __future__ import annotations

import json
import logging
from typing import List, Optional

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
    AutoBoxResponse,
    ClassesResponse,
    CustomItemDeleteResponse,
    CustomItemOut,
    CustomItemsListResponse,
    DetectionResponse,
    EmailMetricsRequest,
    EmailMetricsResponse,
    ErrorResponse,
    HealthResponse,
    ImageDimensions,
    ResetTrackResponse,
)
from app.services.custom_detector import (
    CustomObjectDetector,
    _decode_bgr_image,
    auto_detect_foreground_bbox,
)
from app.services.detector import YOLOObjectDetector
from app.services.email_service import send_metrics_email

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


def get_custom_detector(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> CustomObjectDetector:
    """Retrieve or lazily initialize the CustomObjectDetector service."""
    custom_detector: Optional[CustomObjectDetector] = getattr(
        request.app.state, "custom_detector", None
    )
    if custom_detector is None:
        custom_detector = CustomObjectDetector(
            yoloe_model_path=settings.yoloe_model_path,
            storage_dir=settings.custom_items_dir,
            preferred_device=settings.device,
            default_similarity_threshold=settings.default_similarity_threshold,
        )
        request.app.state.custom_detector = custom_detector
    return custom_detector


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
    "/track/reset",
    response_model=ResetTrackResponse,
    summary="Reset tracker states and object tracking IDs",
)
async def reset_tracker(
    request: Request,
    detector: YOLOObjectDetector = Depends(get_detector),
) -> ResetTrackResponse:
    """Reset the internal ByteTrack tracker states."""
    detector.reset_tracker()
    custom_detector: Optional[CustomObjectDetector] = getattr(
        request.app.state, "custom_detector", None
    )
    if custom_detector is not None:
        custom_detector.reset_tracker()
    return ResetTrackResponse(
        status="ok",
        message="Tracker state successfully reset",
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
    summary="Run real-time YOLO11 and/or custom visual-prompt object detection on an uploaded image frame",
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
    classes: Optional[str] = Query(
        default=None,
        description="Optional comma-separated COCO class names or IDs to detect, e.g. 'person,cup' or '0,41'",
    ),
    classes_form: Optional[str] = Form(
        default=None,
        description="Optional form field for comma-separated COCO classes to detect",
    ),
    custom_items: Optional[str] = Query(
        default=None,
        description="Optional comma-separated custom reference item IDs to detect via YOLOE visual prompting",
    ),
    similarity_threshold: Optional[float] = Query(
        default=None,
        ge=0.1,
        le=1.0,
        description="Visual instance similarity threshold (0.1 to 1.0) for custom reference items",
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

    # Resolve optional target classes filter (query param, form parameter, or multipart field)
    raw_classes = classes if classes is not None else classes_form
    if raw_classes is None:
        try:
            form_data = await request.form()
            form_classes = form_data.get("classes")
            if form_classes is not None and str(form_classes).strip() != "":
                raw_classes = str(form_classes)
        except Exception:
            pass

    resolved_classes: Optional[List[int]] = None
    explicit_no_coco = False
    if raw_classes is not None:
        if raw_classes.strip() == "" or raw_classes.strip() == "__none__":
            explicit_no_coco = True
            resolved_classes = []
        else:
            class_tokens = [c.strip() for c in raw_classes.split(",") if c.strip()]
            resolved_classes = detector.resolve_class_ids(class_tokens)
            if resolved_classes is None and len(class_tokens) > 0:
                resolved_classes = []

    custom_item_ids: List[str] = []
    if custom_items is not None and custom_items.strip() != "":
        custom_item_ids = [cid.strip() for cid in custom_items.split(",") if cid.strip()]

    # Decode image and run YOLO11 and/or custom YOLOE visual-prompt inference
    try:
        if custom_item_ids and explicit_no_coco:
            # Skip YOLO11n entirely when only custom reference items are active
            image_bgr = _decode_bgr_image(contents)
            img_h, img_w = image_bgr.shape[:2]
            custom_detector = get_custom_detector(request, settings)
            custom_dets, custom_ms = custom_detector.detect_bgr(
                image_bgr=image_bgr,
                item_ids=custom_item_ids,
                confidence=resolved_confidence,
                similarity_threshold=similarity_threshold,
            )
            return DetectionResponse(
                image_width=img_w,
                image_height=img_h,
                image_dimensions=ImageDimensions(width=img_w, height=img_h),
                inference_time_ms=custom_ms,
                confidence_threshold=resolved_confidence,
                detection_count=len(custom_dets),
                detections=custom_dets,
            )

        coco_resp = detector.detect(
            image_bytes=contents,
            confidence=resolved_confidence,
            classes=resolved_classes,
        )

        if not custom_item_ids:
            return coco_resp

        image_bgr = _decode_bgr_image(contents)
        custom_detector = get_custom_detector(request, settings)
        custom_dets, custom_ms = custom_detector.detect_bgr(
            image_bgr=image_bgr,
            item_ids=custom_item_ids,
            confidence=resolved_confidence,
            similarity_threshold=similarity_threshold,
        )
        merged = list(coco_resp.detections) + list(custom_dets)
        merged.sort(key=lambda d: d.confidence, reverse=True)
        return DetectionResponse(
            image_width=coco_resp.image_width,
            image_height=coco_resp.image_height,
            image_dimensions=coco_resp.image_dimensions,
            inference_time_ms=round(coco_resp.inference_time_ms + custom_ms, 2),
            confidence_threshold=coco_resp.confidence_threshold,
            detection_count=len(merged),
            detections=merged,
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


@router.post(
    "/metrics/email",
    response_model=EmailMetricsResponse,
    summary="Email structured session metrics report",
)
async def email_metrics(
    payload: EmailMetricsRequest,
    settings: Settings = Depends(get_settings),
) -> EmailMetricsResponse:
    """Send structured session metrics report via email."""
    recipient = (
        (payload.recipient_email or "").strip()
        or (settings.default_recipient_email or "").strip()
    )
    if not recipient:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Recipient email address is required. Please specify it in the request or set DEFAULT_RECIPIENT_EMAIL in backend/.env.",
        )

    try:
        send_metrics_email(
            recipient=recipient,
            metrics=payload.metrics,
            settings=settings,
        )
        return EmailMetricsResponse(
            status="sent",
            message=f"Metrics report successfully sent to {recipient}",
            recipient=recipient,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Failed to send metrics email.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to dispatch email: {exc}",
        ) from exc


@router.get(
    "/custom-items",
    response_model=CustomItemsListResponse,
    summary="List registered custom reference items",
)
async def list_custom_items(
    custom_detector: CustomObjectDetector = Depends(get_custom_detector),
) -> CustomItemsListResponse:
    """Return all saved custom visual-prompt reference items."""
    items = custom_detector.list_items()
    return CustomItemsListResponse(
        count=len(items),
        items=items,
        yoloe_ready=custom_detector.is_ready,
        device=custom_detector.device,
    )


def _parse_bboxes_form(raw_bboxes: Optional[str]) -> Optional[List[Optional[List[float]]]]:
    """Parse optional JSON array of normalized [x1, y1, x2, y2] boxes."""
    if not raw_bboxes or not raw_bboxes.strip():
        return None
    try:
        parsed = json.loads(raw_bboxes)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid 'bboxes' JSON payload.",
        ) from exc
    if not isinstance(parsed, list):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="'bboxes' must be a JSON array of [x1, y1, x2, y2] boxes.",
        )
    return parsed


@router.post(
    "/custom-items/auto-box",
    response_model=AutoBoxResponse,
    summary="Auto-detect a tight foreground bounding box around the main subject in an uploaded photo",
)
async def auto_box_custom_image(
    file: UploadFile = File(..., description="JPEG, PNG, or WebP reference photo"),
    settings: Settings = Depends(get_settings),
) -> AutoBoxResponse:
    """Estimate a tight foreground box using OpenCV GrabCut to exclude background clutter."""
    content_type = (file.content_type or "").lower().strip()
    if content_type and content_type not in settings.allowed_content_types:
        await file.close()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{content_type}'. Allowed: JPEG, PNG, WebP.",
        )
    raw = await file.read()
    await file.close()
    if not raw:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded image is empty.",
        )
    if len(raw) > settings.max_upload_size_bytes:
        raise HTTPException(status_code=413, detail="Uploaded image exceeds size limit.")
    try:
        return auto_detect_foreground_bbox(raw)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post(
    "/custom-items",
    response_model=CustomItemOut,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new custom reference item from uploaded reference photos",
)
async def create_custom_item(
    name: str = Form(..., description="Display name for the custom item (e.g. 'My Chair')"),
    files: List[UploadFile] = File(..., description="JPEG/PNG/WebP reference photos"),
    bboxes: Optional[str] = Form(
        default=None,
        description="Optional JSON array of normalized [x1, y1, x2, y2] foreground boxes matching files",
    ),
    custom_detector: CustomObjectDetector = Depends(get_custom_detector),
    settings: Settings = Depends(get_settings),
) -> CustomItemOut:
    """Upload reference photo(s) of a specific item and index its visual prompt embeddings."""
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please upload at least one reference image.",
        )

    parsed_bboxes = _parse_bboxes_form(bboxes)
    image_bytes_list: List[bytes] = []
    for f in files:
        content_type = (f.content_type or "").lower().strip()
        if content_type and content_type not in settings.allowed_content_types:
            await f.close()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type '{content_type}'. Allowed: JPEG, PNG, WebP.",
            )
        raw = await f.read()
        await f.close()
        if not raw:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="One of the uploaded reference images is empty.",
            )
        if len(raw) > settings.max_upload_size_bytes:
            max_mb = round(settings.max_upload_size_bytes / (1024 * 1024), 2)
            raise HTTPException(
                status_code=413,
                detail=f"Reference image exceeds maximum size limit of {max_mb} MB.",
            )
        image_bytes_list.append(raw)

    try:
        return custom_detector.add_item(
            name=name,
            image_bytes_list=image_bytes_list,
            bboxes_norm=parsed_bboxes,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Failed to register custom item '%s'.", name)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to index custom reference item: {exc}",
        ) from exc


@router.post(
    "/custom-items/{item_id}/images",
    response_model=CustomItemOut,
    summary="Add more reference angle photos to an existing custom item",
)
async def add_custom_item_images(
    item_id: str,
    files: List[UploadFile] = File(..., description="Additional JPEG/PNG/WebP reference photos"),
    bboxes: Optional[str] = Form(
        default=None,
        description="Optional JSON array of normalized [x1, y1, x2, y2] foreground boxes matching files",
    ),
    custom_detector: CustomObjectDetector = Depends(get_custom_detector),
    settings: Settings = Depends(get_settings),
) -> CustomItemOut:
    """Add extra angle photos to improve visual matching robustness for an existing custom item."""
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please upload at least one reference image.",
        )

    parsed_bboxes = _parse_bboxes_form(bboxes)
    image_bytes_list: List[bytes] = []
    for f in files:
        content_type = (f.content_type or "").lower().strip()
        if content_type and content_type not in settings.allowed_content_types:
            await f.close()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type '{content_type}'. Allowed: JPEG, PNG, WebP.",
            )
        raw = await f.read()
        await f.close()
        if not raw:
            continue
        if len(raw) > settings.max_upload_size_bytes:
            raise HTTPException(status_code=413, detail="Reference image exceeds size limit.")
        image_bytes_list.append(raw)

    try:
        return custom_detector.add_images_to_item(
            item_id=item_id,
            image_bytes_list=image_bytes_list,
            bboxes_norm=parsed_bboxes,
        )
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to add images to custom item '%s'.", item_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update custom item: {exc}",
        ) from exc


@router.delete(
    "/custom-items/{item_id}",
    response_model=CustomItemDeleteResponse,
    summary="Delete a registered custom reference item",
)
async def delete_custom_item(
    item_id: str,
    custom_detector: CustomObjectDetector = Depends(get_custom_detector),
) -> CustomItemDeleteResponse:
    """Remove a custom item and delete its saved reference images."""
    deleted = custom_detector.delete_item(item_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Custom item '{item_id}' not found.",
        )
    return CustomItemDeleteResponse(
        status="ok",
        message="Custom reference item deleted",
        deleted_id=item_id,
    )
