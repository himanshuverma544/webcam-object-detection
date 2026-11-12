# Multi-platform production image for FastAPI + PyTorch + Ultralytics YOLO11
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=7860 \
    HOST=0.0.0.0

# Install required system packages for OpenCV and compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up user 1000 for Hugging Face Spaces compatibility
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /app

# Upgrade pip and install CPU-only PyTorch for fast, lightweight builds (~180MB)
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and pretrained weights
COPY --chown=user:user . /app

# Ensure non-root user permissions
RUN chown -R user:user /app
USER user

# Hugging Face Spaces routes traffic to port 7860 by default
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
  CMD curl -f http://127.0.0.1:${PORT:-7860}/health || exit 1

# Start FastAPI via Uvicorn bound to 0.0.0.0 and $PORT
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
