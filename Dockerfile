# ─────────────────────────────────────────────────────────────
# Dockerfile — DreamForge (Stable Diffusion v1.5 Web Panel)
# Модель запекается в образ при сборке → не нужен интернет в runtime
# ─────────────────────────────────────────────────────────────

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# HuggingFace cache
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV DIFFUSERS_CACHE=/app/.cache/huggingface

# Модель запечена в образ — runtime не качает из HF
ENV MODEL_ID=runwayml/stable-diffusion-v1-5
ENV MODEL_PATH=/app/models/stable-diffusion-v1-5
ENV MODELS_DIR=/app/models

# HF токен (опционально, для приватных моделей)
# Передаётся как build arg чтобы не попасть в финальный слой
ARG HF_TOKEN=""

# Port
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

# Директории для пользовательских моделей / lora / vae / embeddings
RUN mkdir -p /app/models/loras /app/models/vae /app/models/embeddings

# ── Скачать SD v1.5 в образ при сборке ───────────────────────
# Если HF_TOKEN передан (--build-arg HF_TOKEN=...), используем его
RUN python3 - <<'PYEOF'
import os, sys
from huggingface_hub import snapshot_download
token = os.environ.get("HF_TOKEN") or None
print("[BUILD] Downloading runwayml/stable-diffusion-v1-5 into image...")
try:
    snapshot_download(
        repo_id="runwayml/stable-diffusion-v1-5",
        local_dir="/app/models/stable-diffusion-v1-5",
        ignore_patterns=["*.msgpack", "*.ot", "flax_model*", "tf_model*", "rust_model*"],
        token=token,
    )
    print("[BUILD] Model downloaded OK")
except Exception as e:
    print(f"[BUILD] WARNING: download failed: {e}", file=sys.stderr)
    print("[BUILD] Container will start but model must be uploaded via UI", file=sys.stderr)
PYEOF

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
