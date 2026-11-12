# Real-Time Webcam Object Detection (YOLO11n + FastAPI + React)

A full-stack real-time computer vision application that captures live video frames from your browser webcam, runs local object detection inference using an **Ultralytics YOLO11n (`yolo11n.pt`)** model in a **Python FastAPI** backend, draws synchronized bounding boxes over the live video stream, and displays real-time telemetry and detected object lists in a responsive **React + Vite** dashboard.

---

## 1. Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Frontend** | React 18, Vite 5, JavaScript (ESModules), HTML5 `<video>` + `<canvas>`, MediaDevices API, Responsive CSS |
| **Backend** | Python 3, FastAPI, Uvicorn, OpenCV (`cv2`), NumPy, Pydantic v2 |
| **AI / Computer Vision** | Ultralytics YOLO11n (`yolo11n.pt`), PyTorch (`torch`, `torchvision`), 80 COCO pretrained classes |

---

## 2. Project Structure

```text
webcam-object-detection/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI app factory, lifespan model loader, CORS, safe exception handlers
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   └── routes.py        # GET /health, GET /classes, POST /detect
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   └── config.py        # Environment variable configuration via Pydantic Settings
│   │   ├── schemas/
│   │   │   ├── __init__.py
│   │   │   └── detection.py     # Structured Pydantic response schemas
│   │   └── services/
│   │       ├── __init__.py
│   │       └── detector.py      # YOLO11n singleton service, device selection, OpenCV decoding
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── conftest.py          # Mocked YOLO model and synthetic OpenCV image fixtures
│   │   └── test_api.py          # Pytest + FastAPI TestClient automated tests
│   ├── requirements.txt
│   ├── .env.example
│   └── README.md
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Header.jsx       # Connection status, device indicator, and detection status pills
│   │   │   ├── VideoViewport.jsx# Live <video> element and ResizeObserver-aligned <canvas> overlay
│   │   │   ├── ControlPanel.jsx # Camera start/stop, detection pause/resume, confidence slider, telemetry
│   │   │   └── DetectionList.jsx# Live detected-object list with confidence bars, bbox coords, COCO drawer
│   │   ├── hooks/
│   │   │   └── useWebcamDetection.js # MediaDevices lifecycle, non-overlapping request loop, error recovery
│   │   ├── services/
│   │   │   └── api.js           # Backend HTTP client (/health, /classes, /detect)
│   │   ├── App.jsx
│   │   ├── App.css
│   │   ├── index.css
│   │   └── main.jsx
│   ├── index.html
│   ├── vite.config.js
│   ├── package.json
│   └── .env.example
├── .gitignore
└── README.md
```

---

## 3. Installation & Running the Application

Use **two separate terminal windows** — one for the Python FastAPI backend and one for the React Vite frontend.

### Step 1: Backend Setup & Startup (Terminal 1)

> **macOS Note (Intel `x86_64` vs Apple Silicon `arm64`):**
> - On **Apple Silicon (`arm64`)** Macs, PyTorch automatically uses MPS GPU acceleration when `DEVICE="auto"` (or falls back to CPU if an operator is unsupported).
> - On **Intel (`x86_64`)** Macs, official PyTorch wheels on PyPI (`torch==2.2.2`) target Python 3.8–3.12 with `numpy<2.0.0`. If your default Homebrew `python3` is Python 3.14, create the virtual environment with `/usr/bin/python3` (macOS Command Line Tools Python 3.9) or Python 3.11/3.12.

```bash
cd backend

# 1. Create a virtual environment
/usr/bin/python3 -m venv .venv   # Or `python3 -m venv .venv` on Apple Silicon / Linux

# 2. Activate the virtual environment
source .venv/bin/activate

# 3. Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Optional: copy environment configuration template
cp -n .env.example .env

# 5. Start the FastAPI server bound to localhost:8000
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
```

#### Model-Weight Download & Initial Startup Expectations
- When the backend starts for the first time, Ultralytics downloads `yolo11n.pt` (~5.4 MB) into `backend/yolo11n.pt` if not already present.
- During startup, the backend runs a lightweight `64×64` warmup pass on the selected compute device (`cuda`, `mps`, or `cpu`).
- `GET /health` will only report `"model_loaded": true` after `yolo11n.pt` has loaded and passed warmup.

---

### Step 2: Frontend Setup & Startup (Terminal 2)

```bash
cd frontend

# 1. Install Node.js dependencies
npm install

# 2. Optional: copy frontend environment variables template
cp -n .env.example .env

# 3. Start the Vite development server on localhost:5173
npm run dev
```

Open **`http://127.0.0.1:5173`** (or **`http://localhost:5173`**) in your browser.

---

## 4. API Endpoints Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Returns API status (`ok` / `degraded`), `model_loaded` boolean, `model_name`, active compute `device`, and `classes_count`. |
| `GET` | `/classes` | Returns all 80 supported COCO class names and their numeric class ID mapping. |
| `POST` | `/detect` | Accepts a `multipart/form-data` image upload (`file`: JPEG, PNG, or WebP) and query parameter `confidence` (`0.01`–`1.0`, default `0.35`). Returns image dimensions, inference latency in ms, and detected objects with `[x1, y1, x2, y2]` pixel coordinates. |
| `GET` | `/docs` | Interactive OpenAPI (Swagger UI) documentation. |

### Example `POST /detect` Response

```json
{
  "image_width": 640,
  "image_height": 480,
  "image_dimensions": {
    "width": 640,
    "height": 480
  },
  "inference_time_ms": 38.42,
  "confidence_threshold": 0.35,
  "detection_count": 2,
  "detections": [
    {
      "label": "person",
      "class_id": 0,
      "confidence": 0.9124,
      "bbox": [112.5, 48.0, 465.2, 472.8]
    },
    {
      "label": "cell phone",
      "class_id": 67,
      "confidence": 0.6418,
      "bbox": [310.0, 220.4, 375.5, 318.9]
    }
  ]
}
```

---

## 5. Testing & Verification

### Run Backend Automated Tests (pytest)
```bash
cd backend
.venv/bin/pytest -v
```

### Run Frontend Production Build
```bash
cd frontend
npm run build
```

---

## 6. Troubleshooting

1. **Camera Permission Denied (macOS & Browser):**
   - Browser webcam access via `navigator.mediaDevices.getUserMedia` requires a secure context (`http://localhost`, `http://127.0.0.1`, or `https://`).
   - On macOS, ensure your browser (Chrome, Safari, Firefox, or Brave) is granted Camera permission under **System Settings → Privacy & Security → Camera**.
   - Click the site permissions icon next to the browser address bar and set **Camera** to **Allow**, then click **Start Camera**.

2. **CORS Errors:**
   - By default, the backend allows `http://localhost:5173`, `http://127.0.0.1:5173`, `http://localhost:4173`, and `http://127.0.0.1:4173`.
   - If Vite selects a different port (e.g., `5174`), add it to `CORS_ORIGINS` in `backend/.env` and restart the backend server.

3. **Python Dependencies / PyTorch Installation on macOS:**
   - If `pip install torch` reports `No matching distribution found for torch` on an Intel Mac, you are likely using Python 3.13/3.14 where x86_64 wheels are unavailable. Recreate `.venv` with `/usr/bin/python3` (Python 3.9) or `python3.11`/`python3.12`.
   - Keep `numpy<2.0.0` when using `torch==2.2.2` on Intel macOS to avoid C-extension ABI incompatibilities.

4. **Model-Loading Errors:**
   - If the first startup fails to download `yolo11n.pt` due to a firewall or offline network, manually download `yolo11n.pt` from the [Ultralytics Assets Releases](https://github.com/ultralytics/assets/releases) and place it inside the `backend/` folder, or set `YOLO_MODEL_PATH=/path/to/yolo11n.pt` in `backend/.env`.
