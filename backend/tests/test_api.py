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


class _FakeCustomDetector:
    """Lightweight in-memory stub of CustomObjectDetector for fast API tests."""

    def __init__(self) -> None:
        from app.schemas.detection import CustomItemOut

        self.is_ready = True
        self.device = "cpu"
        self._items = {}
        self._CustomItemOut = CustomItemOut

    def list_items(self):
        return list(self._items.values())

    def add_item(self, name: str, image_bytes_list, bboxes_norm=None):
        if not name.strip():
            raise ValueError("Item name cannot be empty.")
        iid = f"item_{len(self._items) + 1}"
        item = self._CustomItemOut(
            id=iid,
            name=name.strip(),
            class_id=1000 + len(self._items),
            image_count=len(image_bytes_list),
            thumbnail_base64="data:image/jpeg;base64,/9j/4AAQ",
            created_at="2026-11-18T00:00:00+00:00",
        )
        self._items[iid] = item
        self.last_bboxes = bboxes_norm
        return item

    def add_images_to_item(self, item_id: str, image_bytes_list, bboxes_norm=None):
        if item_id not in self._items:
            raise KeyError(f"Custom item '{item_id}' not found.")
        old = self._items[item_id]
        updated = old.model_copy(update={"image_count": old.image_count + len(image_bytes_list)})
        self._items[item_id] = updated
        self.last_bboxes = bboxes_norm
        return updated

    def delete_item(self, item_id: str) -> bool:
        return self._items.pop(item_id, None) is not None

    def reset_tracker(self) -> None:
        pass

    def unload(self) -> None:
        pass

    def detect_bgr(self, image_bgr, item_ids, confidence=0.35, similarity_threshold=None):
        from app.schemas.detection import DetectionItem

        out = []
        for iid in item_ids:
            if iid in self._items:
                item = self._items[iid]
                out.append(
                    DetectionItem(
                        label=item.name,
                        class_id=item.class_id,
                        confidence=0.88,
                        bbox=[10.0, 20.0, 120.0, 180.0],
                        track_id=10001,
                        is_custom=True,
                        custom_item_id=item.id,
                        similarity=0.91,
                    )
                )
        return out, 14.5


def test_custom_items_crud_endpoints(
    client: TestClient, sample_jpeg_bytes: bytes
) -> None:
    """Custom reference items support list, create, add-images, and delete."""
    fake = _FakeCustomDetector()
    client.app.state.custom_detector = fake

    res_list = client.get("/custom-items")
    assert res_list.status_code == 200
    assert res_list.json()["count"] == 0

    res_create = client.post(
        "/custom-items",
        data={"name": "My Office Chair"},
        files=[("files", ("chair.jpg", sample_jpeg_bytes, "image/jpeg"))],
    )
    assert res_create.status_code == 201
    created = res_create.json()
    assert created["name"] == "My Office Chair"
    assert created["image_count"] == 1
    assert created["class_id"] >= 1000

    res_add = client.post(
        f"/custom-items/{created['id']}/images",
        files=[("files", ("chair_side.jpg", sample_jpeg_bytes, "image/jpeg"))],
    )
    assert res_add.status_code == 200
    assert res_add.json()["image_count"] == 2

    res_del = client.delete(f"/custom-items/{created['id']}")
    assert res_del.status_code == 200
    assert res_del.json()["deleted_id"] == created["id"]

    res_del_missing = client.delete("/custom-items/item_nonexistent")
    assert res_del_missing.status_code == 404


def test_detect_custom_only_and_merged_with_coco(
    client: TestClient, sample_jpeg_bytes: bytes
) -> None:
    """POST /detect supports custom-only detection and merged COCO + custom detection."""
    fake = _FakeCustomDetector()
    client.app.state.custom_detector = fake
    created = fake.add_item("My Chair", [sample_jpeg_bytes])

    # Custom-only (classes empty) skips COCO and returns only custom item
    res_custom_only = client.post(
        f"/detect?confidence=0.35&classes=&custom_items={created.id}&similarity_threshold=0.65",
        files={"file": ("webcam.jpg", sample_jpeg_bytes, "image/jpeg")},
    )
    assert res_custom_only.status_code == 200
    data_only = res_custom_only.json()
    assert data_only["detection_count"] == 1
    assert data_only["detections"][0]["label"] == "My Chair"
    assert data_only["detections"][0]["is_custom"] is True
    assert data_only["detections"][0]["custom_item_id"] == created.id
    assert data_only["detections"][0]["similarity"] == 0.91

    # Merged COCO (person) + Custom (My Chair)
    res_merged = client.post(
        f"/detect?confidence=0.35&classes=person&custom_items={created.id}",
        files={"file": ("webcam.jpg", sample_jpeg_bytes, "image/jpeg")},
    )
    assert res_merged.status_code == 200
    data_merged = res_merged.json()
    assert data_merged["detection_count"] == 2
    labels = {d["label"] for d in data_merged["detections"]}
    assert labels == {"person", "My Chair"}


def test_auto_box_endpoint_and_custom_bboxes_form(
    client: TestClient, sample_jpeg_bytes: bytes
) -> None:
    """POST /custom-items/auto-box returns valid normalized bbox and POST /custom-items accepts bboxes JSON."""
    res_auto = client.post(
        "/custom-items/auto-box",
        files={"file": ("wallet.jpg", sample_jpeg_bytes, "image/jpeg")},
    )
    assert res_auto.status_code == 200
    auto_data = res_auto.json()
    assert len(auto_data["bbox_norm"]) == 4
    x1, y1, x2, y2 = auto_data["bbox_norm"]
    assert 0.0 <= x1 < x2 <= 1.0
    assert 0.0 <= y1 < y2 <= 1.0

    fake = _FakeCustomDetector()
    client.app.state.custom_detector = fake
    res_create = client.post(
        "/custom-items",
        data={"name": "My Wallet", "bboxes": "[[0.23, 0.32, 0.78, 0.71]]"},
        files=[("files", ("wallet.jpg", sample_jpeg_bytes, "image/jpeg"))],
    )
    assert res_create.status_code == 201
    assert fake.last_bboxes == [[0.23, 0.32, 0.78, 0.71]]


def test_chat_endpoint_camera_and_confidence(client: TestClient) -> None:
    """POST /chat correctly processes camera and confidence tuning commands."""
    res = client.post("/chat", json={"message": "start the camera and set confidence to 70%"})
    assert res.status_code == 200
    data = res.json()
    action_types = [a["type"] for a in data["actions"]]
    assert "START_CAMERA" in action_types
    assert "SET_CONFIDENCE" in action_types
    conf_act = next(a for a in data["actions"] if a["type"] == "SET_CONFIDENCE")
    assert round(conf_act["payload"]["value"], 2) == 0.70


def test_chat_endpoint_coco_classes(client: TestClient) -> None:
    """POST /chat filters COCO classes, selects all, and clears classes."""
    res1 = client.post("/chat", json={"message": "only detect people, cups, and cell phones"})
    assert res1.status_code == 200
    act1 = res1.json()["actions"][0]
    assert act1["type"] == "SET_COCO_CLASSES"
    assert "person" in act1["payload"]["classes"]
    assert "cup" in act1["payload"]["classes"]
    assert "cell phone" in act1["payload"]["classes"]

    res_all = client.post("/chat", json={"message": "enable all classes"})
    assert res_all.status_code == 200
    act_all = res_all.json()["actions"][0]
    assert act_all["payload"]["mode"] == "select_all"
    assert len(act_all["payload"]["classes"]) == 80

    res_clear = client.post("/chat", json={"message": "clear all classes"})
    assert res_clear.status_code == 200
    act_clear = res_clear.json()["actions"][0]
    assert act_clear["payload"]["mode"] == "clear_all"
    assert len(act_clear["payload"]["classes"]) == 0


def test_chat_endpoint_guard_and_metrics(client: TestClient) -> None:
    """POST /chat correctly handles scene guarding and metrics."""
    res_guard = client.post("/chat", json={"message": "guard the scene right now"})
    assert res_guard.status_code == 200
    assert any(a["type"] == "GUARD_SCENE" and a["payload"]["arm"] is True for a in res_guard.json()["actions"])

    res_metrics = client.post("/chat", json={"message": "show session metrics summary"})
    assert res_metrics.status_code == 200
    assert any(a["type"] == "SHOW_METRICS" for a in res_metrics.json()["actions"])







