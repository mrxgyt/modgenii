#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# deploy.sh — Deploy DreamForge to Google Cloud Run (GPU)
# Usage: bash deploy.sh
# ─────────────────────────────────────────────────────────────
set -euo pipefail

# ══════════════════════════════════════
# CONFIG — Edit these before deploying
# ══════════════════════════════════════
PROJECT_ID="your-gcp-project-id"           # ← change this
REGION="us-central1"                        # GPU supported region
SERVICE_NAME="dreamforge"
REPO_NAME="dreamforge-repo"                 # Artifact Registry repo name
IMAGE_NAME="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${SERVICE_NAME}:latest"

# SD model (can change to any SD1.5-compatible model on HuggingFace)
MODEL_ID="runwayml/stable-diffusion-v1-5"

# Optional: Set HuggingFace token if model is gated
HF_TOKEN=""   # leave empty if not needed

# Cost controls
MAX_INSTANCES=1      # prevent runaway costs
MIN_INSTANCES=0      # scale to zero when idle
REQUEST_TIMEOUT=300  # seconds (5 min for slow GPU / CPU)

# ══════════════════════════════════════
# 1. Authenticate & set project
# ══════════════════════════════════════
echo "🔐 Setting project to ${PROJECT_ID}..."
gcloud config set project "${PROJECT_ID}"

# ══════════════════════════════════════
# 2. Enable required APIs
# ══════════════════════════════════════
echo "⚙️  Enabling APIs..."
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  --quiet

# ══════════════════════════════════════
# 3. Create Artifact Registry repo (if not exists)
# ══════════════════════════════════════
echo "📦 Creating Artifact Registry repo..."
gcloud artifacts repositories create "${REPO_NAME}" \
  --repository-format=docker \
  --location="${REGION}" \
  --quiet 2>/dev/null || echo "   (repo already exists)"

gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

# ══════════════════════════════════════
# 4. Build & push Docker image
# ══════════════════════════════════════
echo "🔨 Building Docker image (this may take 10-15 minutes first time)..."
docker build -t "${IMAGE_NAME}" .

echo "⬆️  Pushing image to Artifact Registry..."
docker push "${IMAGE_NAME}"

# ══════════════════════════════════════
# 5. Deploy to Cloud Run with GPU
# ══════════════════════════════════════
echo "🚀 Deploying to Cloud Run with NVIDIA L4 GPU..."

SET_ENV="MODEL_ID=${MODEL_ID}"
if [ -n "${HF_TOKEN}" ]; then
  SET_ENV="${SET_ENV},HF_TOKEN=${HF_TOKEN}"
fi

gcloud run deploy "${SERVICE_NAME}" \
  --image "${IMAGE_NAME}" \
  --region "${REGION}" \
  --gpu 1 \
  --gpu-type nvidia-l4 \
  --cpu 8 \
  --memory 32Gi \
  --max-instances "${MAX_INSTANCES}" \
  --min-instances "${MIN_INSTANCES}" \
  --timeout "${REQUEST_TIMEOUT}" \
  --port 8080 \
  --allow-unauthenticated \
  --set-env-vars "${SET_ENV}" \
  --no-cpu-throttling \
  --quiet

# ══════════════════════════════════════
# 6. Print service URL
# ══════════════════════════════════════
echo ""
echo "✅ Deployment complete!"
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
  --region="${REGION}" --format='value(status.url)')
echo "🌐 Service URL: ${SERVICE_URL}"
echo ""
echo "   Open in browser: ${SERVICE_URL}"
echo "   Health check:    ${SERVICE_URL}/health"
echo "   API status:      ${SERVICE_URL}/api/status"
echo ""
echo "⚠️  NOTE: First request will be slow (~30-60s) while the model downloads."
echo "   Subsequent requests should generate in 5-15 seconds on L4 GPU."
