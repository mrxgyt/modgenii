import os
import io
import base64
import asyncio
import logging
import time
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, AutoencoderKL
from PIL import Image

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Directories
# ──────────────────────────────────────────────
MODELS_DIR     = Path(os.getenv("MODELS_DIR", "/app/models"))
LORAS_DIR      = MODELS_DIR / "loras"
VAE_DIR        = MODELS_DIR / "vae"
EMBEDDINGS_DIR = MODELS_DIR / "embeddings"

for d in [MODELS_DIR, LORAS_DIR, VAE_DIR, EMBEDDINGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTS = {".safetensors", ".ckpt", ".pt", ".bin"}

# ──────────────────────────────────────────────
# Global state
# ──────────────────────────────────────────────
MODEL_PATH  = os.getenv("MODEL_PATH", "").strip()
MODEL_ID    = os.getenv("MODEL_ID", "runwayml/stable-diffusion-v1-5")
HF_TOKEN    = os.getenv("HF_TOKEN", None)
UPLOAD_CHUNK = 1024 * 1024  # 1 MB

pipeline:          Optional[StableDiffusionPipeline] = None
model_status:      str = "waiting"   # waiting | loading | ready | error
model_error:       str = ""
current_model:     str = ""
active_lora:       str = ""
active_vae:        str = ""
active_embeddings: List[str] = []
device: str = "cuda" if torch.cuda.is_available() else "cpu"
generation_lock = asyncio.Lock()
model_load_lock = asyncio.Lock()

# ── Event log (отображается на веб-панели) ──────────────────
model_log: List[dict] = []          # [{ts, level, msg}, ...]
MAX_LOG   = 120                     # хранить не более N записей


def log_event(level: str, msg: str):
    """level: info | ok | warn | error"""
    entry = {"ts": time.strftime("%H:%M:%S"), "level": level, "msg": msg}
    model_log.append(entry)
    if len(model_log) > MAX_LOG:
        model_log.pop(0)
    log_fn = logger.info if level in ("info", "ok") else \
              logger.warning if level == "warn" else logger.error
    log_fn(msg)


# ──────────────────────────────────────────────
# Model loading helpers
# ──────────────────────────────────────────────
def _build_pipeline(source: str, dtype) -> StableDiffusionPipeline:
    """Load pipeline from local path (file or dir) or HF hub."""
    p = Path(source)
    if p.exists():
        if p.is_file():
            size_gb = round(p.stat().st_size / 1e9, 2)
            log_event("info", f"📂 Файл найден: {p.name} ({size_gb} GB)")
            log_event("info", "⚙️  Десериализация весов модели (может занять несколько минут)...")
            return StableDiffusionPipeline.from_single_file(
                str(p), torch_dtype=dtype,
                safety_checker=None, requires_safety_checker=False,
            )
        else:
            log_event("info", f"📁 Директория модели: {p.name}")
            log_event("info", "⚙️  Загрузка из директории...")
            return StableDiffusionPipeline.from_pretrained(
                str(p), torch_dtype=dtype,
                safety_checker=None, requires_safety_checker=False,
                local_files_only=True,
            )
    else:
        log_event("info", f"🌐 Скачивание из HuggingFace: {source}")
        return StableDiffusionPipeline.from_pretrained(
            source, torch_dtype=dtype,
            safety_checker=None, requires_safety_checker=False,
            token=HF_TOKEN,
        )


def _apply_device(pipe: StableDiffusionPipeline) -> StableDiffusionPipeline:
    if device == "cuda":
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_name = torch.cuda.get_device_name(0)
        pipe = pipe.to("cuda")
        pipe.enable_attention_slicing()
        log_event("ok", f"⚡ GPU: {gpu_name}  |  VRAM: {vram:.1f} GB")
        log_event("ok", "✅ Attention slicing включён (экономия VRAM)")
    else:
        pipe.enable_attention_slicing(1)
        log_event("warn", "💻 CPU режим — генерация будет медленной (5-20 мин/изображение)")
    return pipe


def load_model(source: str = ""):
    global pipeline, model_status, model_error, current_model, active_lora, active_vae, active_embeddings
    model_status = "loading"
    model_error  = ""
    t0 = time.time()
    try:
        if not source:
            if MODEL_PATH and Path(MODEL_PATH).exists():
                source = MODEL_PATH
            elif MODEL_ID:
                source = MODEL_ID
            else:
                model_status = "waiting"
                log_event("info", "⏳ Ожидание — загрузите файл модели через вкладку «Модели»")
                return

        log_event("info", f"🚀 Начало загрузки: {Path(source).name}")
        log_event("info", f"🖥️  Устройство: {'CUDA (GPU)' if device == 'cuda' else 'CPU'}")
        log_event("info", f"🔢 Точность: {'float16 (быстрее)' if device == 'cuda' else 'float32 (CPU)'} ")

        dtype = torch.float16 if device == "cuda" else torch.float32
        pipe  = _build_pipeline(source, dtype)

        log_event("info", "📅 Настройка планировщика DPMSolver++...")
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

        log_event("info", f"📦 Перенос модели на {device.upper()}...")
        pipe = _apply_device(pipe)

        pipeline          = pipe
        current_model     = str(source)
        active_lora       = ""
        active_vae        = ""
        active_embeddings = []
        model_status      = "ready"
        elapsed = round(time.time() - t0, 1)
        log_event("ok", f"✅ Модель готова! Время инициализации: {elapsed}с")
        log_event("ok", f"🎨 Можно генерировать изображения")
    except Exception as exc:
        model_status = "error"
        model_error  = str(exc)
        elapsed = round(time.time() - t0, 1)
        log_event("error", f"❌ Ошибка загрузки ({elapsed}с): {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Запускаем авто-загрузку только если модель уже есть локально или задан MODEL_ID
    if MODEL_PATH and Path(MODEL_PATH).exists():
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, load_model, MODEL_PATH)
    elif MODEL_ID:
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, load_model, MODEL_ID)
    else:
        logger.info("No MODEL_PATH or MODEL_ID set — waiting for user to upload a model.")
        # model_status остаётся 'waiting'
    yield

app = FastAPI(title="DreamForge", lifespan=lifespan)


# ──────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────
class GenerateRequest(BaseModel):
    prompt:          str   = Field(..., min_length=1, max_length=1000)
    negative_prompt: str   = Field(
        default="blurry, bad anatomy, bad hands, extra limbs, deformed, ugly, lowres, text, watermark",
        max_length=1000)
    steps:           int   = Field(default=25, ge=1,   le=50)
    cfg_scale:       float = Field(default=7.5, ge=1.0, le=20.0)
    width:           int   = Field(default=512, ge=256, le=768)
    height:          int   = Field(default=512, ge=256, le=768)
    seed:            int   = Field(default=-1,  ge=-1,  le=2147483647)
    lora_strength:   float = Field(default=0.8, ge=0.0, le=1.5)


class LoadModelRequest(BaseModel):
    filename: str   # filename inside MODELS_DIR, or "huggingface" for default


class LoadLoraRequest(BaseModel):
    filename: str   # filename inside LORAS_DIR, or "" to unload
    strength: float = 0.8


class LoadVaeRequest(BaseModel):
    filename: str   # filename inside VAE_DIR, or "" to unload


class LoadEmbeddingRequest(BaseModel):
    filename: str   # filename inside EMBEDDINGS_DIR


# ──────────────────────────────────────────────
# Status & health
# ──────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/status")
async def get_status():
    return {
        "status":            model_status,   # waiting | loading | ready | error
        "device":            device,
        "gpu_available":     torch.cuda.is_available(),
        "gpu_name":          torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "current_model":     Path(current_model).name if current_model else None,
        "active_lora":       active_lora,
        "active_vae":        active_vae,
        "active_embeddings": active_embeddings,
        "error":             model_error if model_status == "error" else None,
    }


@app.get("/api/log")
async def get_log():
    """Возвращает лог событий загрузки модели для отображения на веб-панели."""
    return {"log": model_log}


@app.post("/api/log/clear")
async def clear_log():
    model_log.clear()
    return {"ok": True}


# ──────────────────────────────────────────────
# File listing
# ──────────────────────────────────────────────
def _list_files(directory: Path) -> List[dict]:
    result = []
    if directory.exists():
        for f in sorted(directory.iterdir()):
            if f.is_file() and f.suffix.lower() in ALLOWED_EXTS:
                result.append({"name": f.name, "size_mb": round(f.stat().st_size / 1e6, 1)})
    return result


@app.get("/api/models/list")
async def list_models():
    return {"models": _list_files(MODELS_DIR)}


@app.get("/api/loras/list")
async def list_loras():
    return {"loras": _list_files(LORAS_DIR)}


@app.get("/api/vae/list")
async def list_vae():
    return {"vaes": _list_files(VAE_DIR)}


@app.get("/api/embeddings/list")
async def list_embeddings():
    return {"embeddings": _list_files(EMBEDDINGS_DIR)}


# ──────────────────────────────────────────────
# File upload  (streaming, works for large files)
# ──────────────────────────────────────────────
@app.post("/api/upload/{model_type}")
async def upload_file(model_type: str, file: UploadFile = File(...)):
    """
    model_type: model | lora | vae | embedding
    Streams the upload directly to disk — handles GB-sized files.
    """
    type_map = {
        "model":     MODELS_DIR,
        "lora":      LORAS_DIR,
        "vae":       VAE_DIR,
        "embedding": EMBEDDINGS_DIR,
    }
    if model_type not in type_map:
        raise HTTPException(400, detail=f"Unknown type '{model_type}'")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTS:
        raise HTTPException(400, detail=f"Unsupported format '{suffix}'. Use: {ALLOWED_EXTS}")

    dest_dir  = type_map[model_type]
    dest_path = dest_dir / file.filename

    try:
        written = 0
        with open(dest_path, "wb") as f:
            while chunk := await file.read(UPLOAD_CHUNK):
                f.write(chunk)
                written += len(chunk)
        size_mb = round(written / 1e6, 1)
        logger.info(f"Uploaded {model_type}: {file.filename} ({size_mb} MB)")
        return {"ok": True, "filename": file.filename, "size_mb": size_mb}
    except Exception as exc:
        if dest_path.exists():
            dest_path.unlink()
        raise HTTPException(500, detail=str(exc))


# ──────────────────────────────────────────────
# Model / LoRA / VAE switching
# ──────────────────────────────────────────────
@app.post("/api/models/load")
async def switch_model(req: LoadModelRequest):
    global model_status
    if model_load_lock.locked():
        raise HTTPException(409, detail="A model is already being loaded")
    async with model_load_lock:
        if req.filename.lower() == "huggingface":
            source = MODEL_ID
        else:
            path = MODELS_DIR / req.filename
            if not path.exists():
                raise HTTPException(404, detail=f"File not found: {req.filename}")
            source = str(path)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, load_model, source)
    return {"ok": True, "loaded": req.filename, "status": model_status}


@app.post("/api/loras/load")
async def load_lora(req: LoadLoraRequest):
    global active_lora
    _require_ready()
    if not req.filename:
        # Unload LoRA
        try:
            pipeline.unload_lora_weights()
        except Exception:
            pass
        active_lora = ""
        return {"ok": True, "active_lora": ""}

    path = LORAS_DIR / req.filename
    if not path.exists():
        raise HTTPException(404, detail=f"LoRA not found: {req.filename}")

    loop = asyncio.get_event_loop()
    def _load():
        try:
            pipeline.unload_lora_weights()
        except Exception:
            pass
        pipeline.load_lora_weights(str(path))
    await loop.run_in_executor(None, _load)
    active_lora = req.filename
    return {"ok": True, "active_lora": active_lora}


@app.post("/api/vae/load")
async def load_vae(req: LoadVaeRequest):
    global active_vae
    _require_ready()
    if not req.filename:
        # Reset VAE to default
        active_vae = ""
        return {"ok": True, "active_vae": ""}

    path = VAE_DIR / req.filename
    if not path.exists():
        raise HTTPException(404, detail=f"VAE not found: {req.filename}")

    dtype = torch.float16 if device == "cuda" else torch.float32
    loop  = asyncio.get_event_loop()
    def _load():
        vae = AutoencoderKL.from_single_file(str(path), torch_dtype=dtype) \
              if path.suffix.lower() in {".safetensors", ".ckpt"} \
              else AutoencoderKL.from_pretrained(str(path), torch_dtype=dtype)
        if device == "cuda":
            vae = vae.to("cuda")
        pipeline.vae = vae
    await loop.run_in_executor(None, _load)
    active_vae = req.filename
    return {"ok": True, "active_vae": active_vae}


# ──────────────────────────────────────────────
# Embeddings (Textual Inversion)
# ──────────────────────────────────────────────
@app.post("/api/embeddings/load")
async def load_embedding(req: LoadEmbeddingRequest):
    """Load a Textual Inversion embedding and register its token."""
    global active_embeddings
    _require_ready()

    path = EMBEDDINGS_DIR / req.filename
    if not path.exists():
        raise HTTPException(404, detail=f"Embedding not found: {req.filename}")

    token = path.stem  # e.g. "bad-hands-5" for bad-hands-5.pt
    loop  = asyncio.get_event_loop()

    def _load():
        pipeline.load_textual_inversion(str(path), token=token)

    await loop.run_in_executor(None, _load)
    if req.filename not in active_embeddings:
        active_embeddings.append(req.filename)
    logger.info(f"Embedding loaded: {req.filename}  token=<{token}>")
    return {"ok": True, "active_embeddings": active_embeddings, "token": token}


@app.post("/api/embeddings/unload")
async def unload_embedding(req: LoadEmbeddingRequest):
    """Unload a specific embedding (removes token, reloads the rest)."""
    global active_embeddings
    _require_ready()

    if req.filename in active_embeddings:
        active_embeddings.remove(req.filename)

    remaining = list(active_embeddings)
    loop = asyncio.get_event_loop()

    def _reload():
        # Diffusers 0.25+: unload_textual_inversion() clears all
        try:
            pipeline.unload_textual_inversion()
        except Exception:
            pass
        for fname in remaining:
            p = EMBEDDINGS_DIR / fname
            if p.exists():
                pipeline.load_textual_inversion(str(p), token=p.stem)

    await loop.run_in_executor(None, _reload)
    logger.info(f"Embedding unloaded: {req.filename}  remaining={remaining}")
    return {"ok": True, "active_embeddings": active_embeddings}


@app.post("/api/embeddings/unload-all")
async def unload_all_embeddings():
    """Clear every active Textual Inversion embedding."""
    global active_embeddings
    _require_ready()
    loop = asyncio.get_event_loop()

    def _clear():
        try:
            pipeline.unload_textual_inversion()
        except Exception:
            pass

    await loop.run_in_executor(None, _clear)
    active_embeddings = []
    logger.info("All embeddings unloaded")
    return {"ok": True, "active_embeddings": []}


def _require_ready():
    if model_status != "ready" or pipeline is None:
        raise HTTPException(503, detail="Model not ready")


# ──────────────────────────────────────────────
# Generate
# ──────────────────────────────────────────────
@app.post("/api/generate")
async def generate(req: GenerateRequest):
    _require_ready()
    if model_status == "loading":
        raise HTTPException(503, detail="Model is still loading...")
    if generation_lock.locked():
        raise HTTPException(429, detail="Another generation is in progress, please wait...")

    async with generation_lock:
        try:
            t_start     = time.time()
            actual_seed = req.seed if req.seed != -1 else int(time.time() * 1000) % 2147483647
            generator   = torch.Generator(device=device).manual_seed(actual_seed)

            logger.info(f"Generating | {req.prompt[:60]!r} | {req.steps}steps cfg={req.cfg_scale} "
                        f"{req.width}x{req.height} seed={actual_seed}")

            kwargs = dict(
                prompt=req.prompt, negative_prompt=req.negative_prompt,
                num_inference_steps=req.steps, guidance_scale=req.cfg_scale,
                width=req.width, height=req.height, generator=generator,
            )
            if active_lora:
                kwargs["cross_attention_kwargs"] = {"scale": req.lora_strength}

            loop   = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: pipeline(**kwargs))

            image: Image.Image = result.images[0]
            buf   = io.BytesIO()
            image.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode()

            elapsed = round(time.time() - t_start, 2)
            logger.info(f"Done in {elapsed}s  seed={actual_seed}")
            return {
                "image_base64": img_b64,
                "seed_used":    actual_seed,
                "time_seconds": elapsed,
                "device":       device,
                "model":        Path(current_model).name if current_model else MODEL_ID,
                "lora":         active_lora,
                "vae":          active_vae,
            }
        except Exception as exc:
            logger.error(f"Generation error: {exc}", exc_info=True)
            raise HTTPException(500, detail=str(exc))


# ──────────────────────────────────────────────
# Static files — mount LAST
# ──────────────────────────────────────────────
app.mount("/", StaticFiles(directory="static", html=True), name="static")
