from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app
from app.services.detector import YOLOObjectDetector


def test_health_endpoint_when_model_ready(client: TestClient) -> None:
    """GET /health reports status='ok' and model_loaded=True when model is loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True
    assert payload["model_ready"] is True
    assert payload["model_name"] == "yolo11n.pt"
    assert payload["device"] == "cpu"
    assert payload["classes_count"] == 80
    assert payload["error"] is None


def test_health_and_detect_when_model_not_loaded(sample_jpeg_bytes: bytes) -> None:
    """GET /health does not report model as ready before loading, and POST /detect returns 503."""
    unloaded_detector = YOLOObjectDetector(model_path="yolo11n.pt")
    unloaded_detector.model = None
    unloaded_detector.model_loaded = False
    unloaded_detector.load_error = "Model still initializing"

    app = create_app()
    app.state.detector = unloaded_detector

    with TestClient(app, raise_server_exceptions=False) as unready_client:
        health_res = unready_client.get("/health")
        assert health_res.status_code == 200
        health_data = health_res.json()
        assert health_data["status"] == "degraded"
        assert health_data["model_loaded"] is False
        assert health_data["model_ready"] is False
        assert health_data["error"] == "Model still initializing"

        detect_res = unready_client.post(
            "/detect",
            files={"file": ("frame.jpg", sample_jpeg_bytes, "image/jpeg")},
        )
        assert detect_res.status_code == 503
        assert "not ready" in detect_res.json()["detail"].lower()


def test_classes_endpoint_returns_coco_classes(client: TestClient) -> None:
    """GET /classes returns all 80 COCO class names and ID mapping."""
    response = client.get("/classes")
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 80
    assert isinstance(payload["classes"], dict)
    assert isinstance(payload["class_names"], list)
    assert len(payload["class_names"]) == 80

    expected_common_objects = {
        "person",
        "bottle",
        "cup",
        "cell phone",
        "laptop",
        "chair",
        "book",
        "car",
    }
    assert expected_common_objects.issubset(set(payload["class_names"]))
    assert payload["classes"]["0"] == "person"
    assert payload["classes"]["63"] == "laptop"


def test_detect_valid_jpeg_schema_and_coordinates(
    client: TestClient, sample_jpeg_bytes: bytes
) -> None:
    """POST /detect returns structured detections with [x1, y1, x2, y2] pixel coordinates."""
    response = client.post(
        "/detect?confidence=0.35",
        files={"file": ("webcam.jpg", sample_jpeg_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()

    assert data["image_width"] == 640
    assert data["image_height"] == 480
    assert data["image_dimensions"] == {"width": 640, "height": 480}
    assert isinstance(data["inference_time_ms"], (int, float))
    assert data["inference_time_ms"] >= 0.0
    assert data["confidence_threshold"] == 0.35
    assert data["detection_count"] == 2
    assert len(data["detections"]) == 2

    first = data["detections"][0]
    assert first["label"] == "person"
    assert first["class_id"] == 0
    assert first["confidence"] == 0.91
    assert first["bbox"] == [50.0, 40.0, 280.0, 420.0]
    assert first["track_id"] == 1

    second = data["detections"][1]
    assert second["label"] == "laptop"
    assert second["class_id"] == 63
    assert second["confidence"] == 0.48
    assert second["bbox"] == [310.0, 200.0, 520.0, 390.0]
    assert second["track_id"] == 2


def test_track_reset_endpoint(client: TestClient) -> None:
    """POST /track/reset resets tracker internal state and returns status ok."""
    response = client.post("/track/reset")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "reset" in data["message"].lower()


def test_detect_supports_png_and_webp(
    client: TestClient, sample_png_bytes: bytes, sample_webp_bytes: bytes
) -> None:
    """POST /detect accepts PNG and WebP image frames."""
    png_res = client.post(
        "/detect",
        files={"file": ("frame.png", sample_png_bytes, "image/png")},
    )
    assert png_res.status_code == 200
    assert png_res.json()["image_dimensions"] == {"width": 320, "height": 240}

    webp_res = client.post(
        "/detect",
        files={"file": ("frame.webp", sample_webp_bytes, "image/webp")},
    )
    assert webp_res.status_code == 200
    assert webp_res.json()["image_dimensions"] == {"width": 320, "height": 240}


def test_detect_rejects_invalid_file_types(client: TestClient) -> None:
    """POST /detect rejects non-image or unsupported MIME types with HTTP 400."""
    text_res = client.post(
        "/detect",
        files={"file": ("notes.txt", b"hello world", "text/plain")},
    )
    assert text_res.status_code == 400
    assert "unsupported file type" in text_res.json()["detail"].lower()

    pdf_res = client.post(
        "/detect",
        files={"file": ("doc.pdf", b"%PDF-1.4...", "application/pdf")},
    )
    assert pdf_res.status_code == 400


def test_detect_rejects_corrupted_or_empty_images(client: TestClient) -> None:
    """POST /detect returns HTTP 400 when uploaded image bytes are empty or corrupted."""
    empty_res = client.post(
        "/detect",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert empty_res.status_code == 400
    assert "empty" in empty_res.json()["detail"].lower()

    corrupted_res = client.post(
        "/detect",
        files={"file": ("corrupt.jpg", b"not-a-valid-jpeg-stream-12345", "image/jpeg")},
    )
    assert corrupted_res.status_code == 400
    assert "invalid or corrupted" in corrupted_res.json()["detail"].lower()


def test_detect_confidence_threshold_filtering_and_validation(
    client: TestClient, sample_jpeg_bytes: bytes
) -> None:
    """Confidence parameter filters detections and rejects out-of-range values with HTTP 422."""
    # Higher confidence (0.75) should filter out the 0.48 laptop detection
    high_conf_res = client.post(
        "/detect?confidence=0.75",
        files={"file": ("frame.jpg", sample_jpeg_bytes, "image/jpeg")},
    )
    assert high_conf_res.status_code == 200
    payload = high_conf_res.json()
    assert payload["confidence_threshold"] == 0.75
    assert payload["detection_count"] == 1
    assert payload["detections"][0]["label"] == "person"

    # Invalid confidence values (< 0.01 or > 1.0 or non-numeric)
    for invalid_conf in ["0.0", "-0.2", "1.5", "not_a_number"]:
        bad_res = client.post(
            f"/detect?confidence={invalid_conf}",
            files={"file": ("frame.jpg", sample_jpeg_bytes, "image/jpeg")},
        )
        assert bad_res.status_code == 422


def test_detect_handles_inference_failure_without_leaking_stack_trace(
    client: TestClient,
    mock_detector: YOLOObjectDetector,
    sample_jpeg_bytes: bytes,
) -> None:
    """When YOLO inference raises an internal exception, API returns 500 without leaking tracebacks."""
    mock_detector.model.should_raise = True  # type: ignore[union-attr]

    response = client.post(
        "/detect",
        files={"file": ("frame.jpg", sample_jpeg_bytes, "image/jpeg")},
    )
    assert response.status_code == 500
    body = response.json()
    assert "detail" in body
    assert "Traceback" not in body["detail"]
    assert "Simulated GPU/CPU tensor execution failure" not in body["detail"]


def test_detect_rejects_oversized_upload(mock_detector: YOLOObjectDetector) -> None:
    """POST /detect returns HTTP 413 when uploaded image exceeds configured size limit."""
    from app.core.config import Settings, get_settings

    app = create_app()
    app.state.detector = mock_detector
    app.dependency_overrides[get_settings] = lambda: Settings(MAX_UPLOAD_SIZE_BYTES=1024)

    with TestClient(app, raise_server_exceptions=False) as limited_client:
        oversized_payload = b"\xff\xd8\xff" + (b"x" * 2048)
        res = limited_client.post(
            "/detect",
            files={"file": ("large.jpg", oversized_payload, "image/jpeg")},
        )
        assert res.status_code == 413
        assert "exceeds maximum size limit" in res.json()["detail"].lower()


def test_email_metrics_unconfigured_smtp(client: TestClient) -> None:
    """POST /metrics/email returns 400 when no email service credentials are configured."""
    from app.core.config import Settings, get_settings

    mock_settings = Settings(
        RESEND_API_KEY=None,
        SMTP_USER=None,
        SMTP_PASSWORD=None,
    )
    client.app.dependency_overrides[get_settings] = lambda: mock_settings

    res = client.post(
        "/metrics/email",
        json={
            "recipient_email": "test-user@example.com",
            "metrics": {
                "timestamp": "2026-11-14 20:20:00",
                "durationFormatted": "5m 20s",
                "frameCount": 240,
            },
        },
    )
    client.app.dependency_overrides.clear()
    assert res.status_code == 400
    assert "not configured" in res.json()["detail"]


def test_email_metrics_success_with_resend_api(
    client: TestClient, monkeypatch
) -> None:
    """POST /metrics/email succeeds when RESEND_API_KEY is configured and Resend API responds 200."""
    from unittest.mock import MagicMock
    import httpx
    from app.core.config import Settings, get_settings

    mock_settings = Settings(
        RESEND_API_KEY="re_test_valid_key_123",
        RESEND_FROM_EMAIL="Webcam Detection <onboarding@resend.dev>",
    )
    client.app.dependency_overrides[get_settings] = lambda: mock_settings

    mock_response = httpx.Response(
        status_code=200,
        json={"id": "email_resend_12345"},
        request=httpx.Request("POST", "https://api.resend.com/emails"),
    )
    mock_client_instance = MagicMock()
    mock_client_instance.__enter__.return_value = mock_client_instance
    mock_client_instance.post.return_value = mock_response

    monkeypatch.setattr(
        "app.services.email_service.httpx.Client",
        lambda *a, **kw: mock_client_instance,
    )

    res = client.post(
        "/metrics/email",
        json={
            "recipient_email": "test-user@example.com",
            "metrics": {
                "timestamp": "2026-11-14 20:20:00",
                "durationFormatted": "5m 20s",
                "frameCount": 240,
                "securityStatus": "CLEAN",
                "topClasses": [{"label": "person", "percent": 100, "count": 240}],
            },
        },
    )
    client.app.dependency_overrides.clear()
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "sent"
    assert data["recipient"] == "test-user@example.com"
    mock_client_instance.post.assert_called_once()


def test_email_metrics_resend_api_error_handling(
    client: TestClient, monkeypatch
) -> None:
    """POST /metrics/email surfaces clear error message when Resend API rejects request."""
    from unittest.mock import MagicMock
    import httpx
    from app.core.config import Settings, get_settings

    mock_settings = Settings(
        RESEND_API_KEY="re_invalid_key",
    )
    client.app.dependency_overrides[get_settings] = lambda: mock_settings

    mock_response = httpx.Response(
        status_code=403,
        json={"message": "Invalid API Key"},
        request=httpx.Request("POST", "https://api.resend.com/emails"),
    )
    mock_client_instance = MagicMock()
    mock_client_instance.__enter__.return_value = mock_client_instance
    mock_client_instance.post.return_value = mock_response

    monkeypatch.setattr(
        "app.services.email_service.httpx.Client",
        lambda *a, **kw: mock_client_instance,
    )

    res = client.post(
        "/metrics/email",
        json={
            "recipient_email": "test-user@example.com",
            "metrics": {
                "timestamp": "2026-11-14 20:20:00",
                "durationFormatted": "5m 20s",
                "frameCount": 240,
            },
        },
    )
    client.app.dependency_overrides.clear()
    assert res.status_code == 400
    assert "Invalid API Key" in res.json()["detail"]


def test_email_metrics_success_with_mocked_smtp(
    client: TestClient, monkeypatch
) -> None:
    """POST /metrics/email succeeds with SMTP fallback when RESEND_API_KEY is not set."""
    from unittest.mock import MagicMock
    from app.core.config import Settings, get_settings

    mock_settings = Settings(
        RESEND_API_KEY=None,
        SMTP_USER="test@example.com",
        SMTP_PASSWORD="testpassword",
        SMTP_HOST="smtp.example.com",
    )
    client.app.dependency_overrides[get_settings] = lambda: mock_settings

    mock_smtp_instance = MagicMock()
    monkeypatch.setattr("smtplib.SMTP", lambda *a, **kw: mock_smtp_instance)

    res = client.post(
        "/metrics/email",
        json={
            "recipient_email": "test-user@example.com",
            "metrics": {
                "timestamp": "2026-11-14 20:20:00",
                "durationFormatted": "5m 20s",
                "frameCount": 240,
                "securityStatus": "CLEAN",
                "topClasses": [{"label": "person", "percent": 100, "count": 240}],
            },
        },
    )
    client.app.dependency_overrides.clear()
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "sent"
    assert data["recipient"] == "test-user@example.com"
    mock_smtp_instance.sendmail.assert_called_once()


def test_email_metrics_uses_default_recipient_from_config(
    client: TestClient, monkeypatch
) -> None:
    """POST /metrics/email falls back to DEFAULT_RECIPIENT_EMAIL if omitted from payload."""
    from unittest.mock import MagicMock
    import httpx
    from app.core.config import Settings, get_settings

    mock_settings = Settings(
        RESEND_API_KEY="re_test_valid_key",
        DEFAULT_RECIPIENT_EMAIL="configured-env@example.com",
    )
    client.app.dependency_overrides[get_settings] = lambda: mock_settings

    mock_response = httpx.Response(
        status_code=200,
        json={"id": "email_resend_999"},
        request=httpx.Request("POST", "https://api.resend.com/emails"),
    )
    mock_client_instance = MagicMock()
    mock_client_instance.__enter__.return_value = mock_client_instance
    mock_client_instance.post.return_value = mock_response

    monkeypatch.setattr(
        "app.services.email_service.httpx.Client",
        lambda *a, **kw: mock_client_instance,
    )

    res = client.post(
        "/metrics/email",
        json={
            "metrics": {
                "timestamp": "2026-11-16 11:30:00",
                "durationFormatted": "3m 15s",
                "frameCount": 180,
            },
        },
    )
    client.app.dependency_overrides.clear()
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "sent"
    assert data["recipient"] == "configured-env@example.com"


def test_detect_with_class_filter_by_name(
    client: TestClient, sample_jpeg_bytes: bytes
) -> None:
    """POST /detect with classes=person returns only person detection, filtering out laptop."""
    response = client.post(
        "/detect?confidence=0.35&classes=person",
        files={"file": ("webcam.jpg", sample_jpeg_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detection_count"] == 1
    assert len(data["detections"]) == 1
    assert data["detections"][0]["label"] == "person"
    assert data["detections"][0]["class_id"] == 0


def test_detect_with_class_filter_by_id(
    client: TestClient, sample_jpeg_bytes: bytes
) -> None:
    """POST /detect with classes=63 returns only laptop detection (class_id=63)."""
    response = client.post(
        "/detect?confidence=0.35&classes=63",
        files={"file": ("webcam.jpg", sample_jpeg_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detection_count"] == 1
    assert len(data["detections"]) == 1
    assert data["detections"][0]["label"] == "laptop"
    assert data["detections"][0]["class_id"] == 63


def test_detect_with_class_filter_multiple(
    client: TestClient, sample_jpeg_bytes: bytes
) -> None:
    """POST /detect with classes=person,laptop returns both matching objects."""
    response = client.post(
        "/detect?confidence=0.35&classes=person,laptop",
        files={"file": ("webcam.jpg", sample_jpeg_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detection_count"] == 2
    labels = {d["label"] for d in data["detections"]}
    assert labels == {"person", "laptop"}


def test_detect_with_class_filter_no_match(
    client: TestClient, sample_jpeg_bytes: bytes
) -> None:
    """POST /detect with classes=bottle returns 0 detections when no bottle exists in frame."""
    response = client.post(
        "/detect?confidence=0.35&classes=bottle",
        files={"file": ("webcam.jpg", sample_jpeg_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detection_count"] == 0
    assert len(data["detections"]) == 0




