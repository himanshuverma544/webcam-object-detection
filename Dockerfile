# Multi-stage Dockerfile: React Frontend + FastAPI AI Backend
# Optimized for Google Cloud Run Free Tier deployment

# --- Stage 1: Build the React + Vite Frontend ---
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
# Build production bundle with relative API endpoints (same origin)
RUN npm run build

# --- Stage 2: Python Backend with PyTorch & YOLO ---
FROM python:3.11-slim AS production

# Install system dependencies required by OpenCV and image decoders
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY backend/requirements.txt ./backend/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r backend/requirements.txt

# Pre-download YOLO11n weights during image build for instant container cold starts
RUN python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')" && \
    mv yolo11n.pt ./backend/yolo11n.pt 2>/dev/null || true

# Copy backend application source
COPY backend/ ./backend/

# Copy built frontend assets from Stage 1 into the container
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Cloud Run defaults: PORT 8080, run on CPU
ENV PORT=8080
ENV HOST=0.0.0.0
ENV DEVICE=cpu
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/backend

WORKDIR /app/backend

EXPOSE 8080

# Run uvicorn bound to Cloud Run's dynamic PORT
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]

