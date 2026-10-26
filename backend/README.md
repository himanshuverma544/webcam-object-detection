# YOLO11 Object Detection Backend (FastAPI + OpenCV + Ultralytics)

This directory contains the Python backend service for the real-time webcam object detection application. It loads the **Ultralytics YOLO11n (`yolo11n.pt`)** pretrained COCO model once at application startup via FastAPI lifespan management and exposes low-latency HTTP endpoints for health checks, class discovery, and frame-by-frame object detection.

## Architecture

```
backend/
  app/
    api/
      routes.py         # GET /health, GET /classes, POST /detect
    core/
      config.py         # Environment variable settings via Pydantic Settings
    schemas/
      detection.py      # Pydantic response & error models
    services/
      detector.py       # YOLO11n model lifecycle, OpenCV decoding, inference
    main.py             # FastAPI app factory, lifespan hook, CORS, exception handlers
  tests/
    conftest.py         # Pytest fixtures & mocked YOLO11 model
    test_api.py         # Automated endpoint & validation test suite
  requirements.txt
  .env.example
```

## Setup & Installation

> **macOS Intel (`x86_64`) Note:** PyTorch publishes prebuilt `macosx_x86_64` wheels for Python 3.8–3.12 (up to `torch 2.2.2`). On Intel Macs where Homebrew `python3` points to Python 3.14, create the virtual environment using `/usr/bin/python3` (Python 3.9) or Python 3.11/3.12 so prebuilt `torch` and `ultralytics` wheels install cleanly. On Apple Silicon (`arm64`) or Linux, standard `python3` works out of the box.

```bash
cd backend

# Create an isolated virtual environment
/usr/bin/python3 -m venv .venv   # Or `python3 -m venv .venv` on Apple Silicon / Linux

# Activate the virtual environment
source .venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Optional: create a local .env file from the example template
cp .env.example .env
```

## Running the Backend Server

Bind to `127.0.0.1:8000` (localhost):

```bash
cd backend
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

- **API Base URL:** `http://127.0.0.1:8000`
- **Interactive Swagger UI:** `http://127.0.0.1:8000/docs`
- **ReDoc Reference:** `http://127.0.0.1:8000/redoc`

### Model-Weight Download Behavior

On first startup, Ultralytics checks if `yolo11n.pt` (~5.4 MB) exists in `backend/`. If not present, it automatically downloads `yolo11n.pt` from the official Ultralytics GitHub assets release and performs a 64×64 warmup pass. Until loading finishes, `GET /health` reports `model_loaded: false`. Once ready, `GET /health` reports `status: "ok"` and `model_loaded: true`.

## Running Automated Tests

The unit test suite mocks the underlying YOLO model so tests run in `< 0.5s` without requiring network access or model weight downloads:

```bash
cd backend
.venv/bin/pytest -v
```
