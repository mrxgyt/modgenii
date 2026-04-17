import os
import io
import base64
import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image

# -----------------------------------------------------
# Logging
# -----------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------
# Global state
# -----------------------------------------------------
# MODEL_PATH - path to local model file (.safetensors / .ckpt)
#              OR path to a diffusers-format directory
#              Set this env var in Northflank to your volume path.
#              Example: /models/v1-5-pruned-emaonly.safetensors
#              If not set - falls back to HuggingFace download.
MODEL_PATH = os.getenv("MODEL_PATH", "").strip()
MODEL_ID   = os.getenv("MODEL_ID", "runwayml/stable-diffusion-v1-5")
HF_TOKEN   = os.getenv("HF_TOKEN", None)

pipeline: Optional[StableDiffusionPipeline] = None
model_status: str = "loading"   # loading | ready | error
model_error:  str = ""
device: str = "cuda" if torch.cuda.is_available() else "cpu"
generation_lock = asyncio.Lock()   # one generation at a time


# -----------------------------------------------------
# Load model
# -----------------------------------------------------
def load_model():
    global pipeline, model_status, model_error, device
    try:
        dtype = torch.float16 if device == "cuda" else torch.float32

        # ---- Determine source ----------------------------------------
        if MODEL_PATH and os.path.exists(MODEL_PATH):

            # Single file: .safetensors or .ckpt
            if os.path.isfile(MODEL_PATH):
                logger.info(f"Loading from local file: {MODEL_PATH}")
                pipe = StableDiffusionPipeline.from_single_file(
                    MODEL_PATH,
                    torch_dtype=dtype,
                    safety_checker=None,
                    requires_safety_checker=False,
                )

            # Diffusers directory format
            else:
                logger.info(f"Loading from local directory: {MODEL_PATH}")
                pipe = StableDiffusionPipeline.from_pretrained(
                    MODEL_PATH,
                    torch_dtype=dtype,
                    safety_checker=None,
                    requires_safety_checker=False,
                    local_files_only=True,
                )

        else:
            if MODEL_PATH:
                logger.warning(
                    f"MODEL_PATH='{MODEL_PATH}' not found on disk. "
                    "Falling back to HuggingFace download."
                )
            logger.info(f"Downloading {MODEL_ID} from HuggingFace...")
            pipe = StableDiffusionPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False,
                token=HF_TOKEN,
            )

        # ---- Scheduler (fast DPM++) ----------------------------------
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config
        )

        # ---- Device --------------------------------------------------
        if device == "cuda":
            pipe = pipe.to("cuda")
            pipe.enable_attention_slicing()
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(
                f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            )
        else:
            # CPU memory optimizations
            pipe.enable_attention_slicing(1)
            logger.warning("CPU mode — each image will take 5-15 minutes!")

        pipeline = pipe
        model_status = "ready"
        logger.info("Model loaded and ready!")

    except Exception as exc:
        model_status = "error"
        model_error = str(exc)
        logger.error(f"Failed to load model: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, load_model)
    yield


# -----------------------------------------------------
# App
# -----------------------------------------------------
app = FastAPI(title="DreamForge — Stable Diffusion Web Panel", lifespan=lifespan)


# -----------------------------------------------------
# Schemas
# -----------------------------------------------------
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000)
    negative_prompt: str = Field(
        default="blurry, bad anatomy, bad hands, extra limbs, deformed, ugly, lowres, text, watermark",
        max_length=1000,
    )
    steps:     int   = Field(default=25,  ge=1,   le=50)
    cfg_scale: float = Field(default=7.5, ge=1.0, le=20.0)
    width:     int   = Field(default=512, ge=256, le=768)
    height:    int   = Field(default=512, ge=256, le=768)
    seed:      int   = Field(default=-1,  ge=-1,  le=2147483647)


class GenerateResponse(BaseModel):
    image_base64: str
    seed_used:    int
    time_seconds: float
    device:       str
    model_source: str


# -----------------------------------------------------
# Routes
# -----------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/status")
async def get_status():
    source = "local file" if (MODEL_PATH and os.path.exists(MODEL_PATH)) else "huggingface"
    return {
        "status":        model_status,
        "device":        device,
        "gpu_available": torch.cuda.is_available(),
        "gpu_name":      torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model_source":  source,
        "model_path":    MODEL_PATH or None,
        "error":         model_error if model_status == "error" else None,
    }


@app.post("/api/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    if model_status == "loading":
        raise HTTPException(status_code=503, detail="Model is still loading, please wait...")
    if model_status == "error":
        raise HTTPException(status_code=500, detail=f"Model failed to load: {model_error}")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if generation_lock.locked():
        raise HTTPException(status_code=429, detail="Another generation is in progress, please wait...")

    async with generation_lock:
        try:
            t_start = time.time()

            actual_seed = req.seed if req.seed != -1 else int(time.time() * 1000) % 2147483647
            generator = torch.Generator(device=device).manual_seed(actual_seed)

            logger.info(
                f"Generating | prompt={req.prompt[:60]!r} | "
                f"steps={req.steps} cfg={req.cfg_scale} "
                f"size={req.width}x{req.height} seed={actual_seed}"
            )

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: pipeline(
                    prompt=req.prompt,
                    negative_prompt=req.negative_prompt,
                    num_inference_steps=req.steps,
                    guidance_scale=req.cfg_scale,
                    width=req.width,
                    height=req.height,
                    generator=generator,
                ),
            )

            image: Image.Image = result.images[0]

            buf = io.BytesIO()
            image.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode()

            elapsed = round(time.time() - t_start, 2)
            source = "local" if (MODEL_PATH and os.path.exists(MODEL_PATH)) else "huggingface"
            logger.info(f"Done in {elapsed}s  seed={actual_seed}")

            return GenerateResponse(
                image_base64=img_b64,
                seed_used=actual_seed,
                time_seconds=elapsed,
                device=device,
                model_source=source,
            )

        except Exception as exc:
            logger.error(f"Generation error: {exc}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))


# -----------------------------------------------------
# Static files — mount LAST
# -----------------------------------------------------
app.mount("/", StaticFiles(directory="static", html=True), name="static")
