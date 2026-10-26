from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router as api_router
from app.core.config import Settings, get_settings
from app.services.detector import YOLOObjectDetector


def configure_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load the YOLO11 model once at application startup and release on shutdown."""
    settings: Settings = get_settings()
    configure_logging(settings.log_level)

    existing_detector = getattr(app.state, "detector", None)
    if existing_detector is None:
        detector = YOLOObjectDetector(
            model_path=settings.yolo_model_path,
            default_confidence=settings.default_confidence,
            iou_threshold=settings.iou_threshold,
            max_detections=settings.max_detections,
            preferred_device=settings.device,
        )
        app.state.detector = detector
        try:
            detector.load_model()
        except Exception as exc:
            logger.error(
                "Startup warning: YOLO model could not be loaded at startup: %s",
                exc,
            )
    else:
        logger.info("Using pre-configured detector instance on app.state.detector.")

    yield

    active_detector = getattr(app.state, "detector", None)
    if active_detector is not None and existing_detector is None:
        active_detector.unload_model()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Real-time webcam object detection backend powered by Ultralytics YOLO11n, "
            "OpenCV, and FastAPI."
        ),
        lifespan=lifespan,
    )

    # Restrict CORS to configured local Vite frontend origins
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @application.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Return clean validation error messages without internal tracebacks."""
        errors = exc.errors()
        messages = []
        for err in errors:
            loc = " -> ".join(str(part) for part in err.get("loc", []))
            msg = err.get("msg", "Invalid value")
            messages.append(f"{loc}: {msg}" if loc else msg)
        detail = "; ".join(messages) if messages else "Invalid request parameters."
        return JSONResponse(
            status_code=422,
            content={"detail": detail},
        )

    @application.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Prevent internal stack traces from leaking to the frontend."""
        logger.exception("Unhandled server exception on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error."},
        )

    application.include_router(api_router)
    return application


app = create_app()
