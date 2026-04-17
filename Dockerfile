# ─────────────────────────────────────────────────────────────
# Dockerfile — DreamForge (Stable Diffusion v1.5 Web Panel)
# Base: NVIDIA CUDA runtime — GPU accelerated on Cloud Run
# ─────────────────────────────────────────────────────────────

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# HuggingFace cache inside container (ephemeral, model re-downloaded on cold start)
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV DIFFUSERS_CACHE=/app/.cache/huggingface

# Model to load (can be overridden via Cloud Run env var)
ENV MODEL_ID=runwayml/stable-diffusion-v1-5

# Port that Cloud Run expects
ENV PORT=8080

# ── System deps ──────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3.11-distutils \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# ── Python deps ───────────────────────────────────────────────
WORKDIR /app

COPY requirements.txt .

# Install PyTorch with CUDA 12.1 support
RUN pip install --no-cache-dir \
    torch==2.3.0 torchvision==0.18.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install the rest
RUN pip install --no-cache-dir -r requirements.txt

# ── App code ─────────────────────────────────────────────────
COPY main.py .
COPY static/ ./static/

# Create cache dir
RUN mkdir -p /app/.cache/huggingface

# ── Healthcheck ───────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# ── Startup ───────────────────────────────────────────────────
EXPOSE 8080

CMD ["python3", "-m", "uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--workers", "1", \
     "--timeout-keep-alive", "300"]
