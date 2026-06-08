#!/usr/bin/env python3
"""Standalone MisoTTS HTTP API server compatible with omnivoice_api.py format.

Endpoints:
  POST /api/misotts/synthesize   Synthesize audio with MisoTTS
  GET  /api/health              Health check
  GET  /api/misotts/status       Model cache status
  POST /api/misotts/unload       Unload model from memory
"""

import argparse
import asyncio
import base64
import hashlib
import json
import logging
import os
import random
import re
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio
from aiohttp import web

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("misotts_api")

ROOT = Path(__file__).resolve().parent
WORK_ROOT = ROOT / "work"
OUTPUT_DIR = WORK_ROOT / "misotts_api_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_TEXT_LEN = int(os.environ.get("MISOTTS_MAX_TEXT_LEN", "2000"))
MAX_REQUEST_MB = int(os.environ.get("MISOTTS_MAX_REQUEST_MB", "64"))
MAX_REQUEST_SIZE = MAX_REQUEST_MB * 1024 * 1024
MISOTTS_SEED_MOD = 2**31 - 1

_API_MODEL = None
_API_MODEL_ID = "MisoLabs/MisoTTS"
_API_DEVICE = None
_API_LOCK = asyncio.Lock()


def get_best_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _json_response(data, status=200):
    return web.json_response(data, status=status)


def _error(message, status=400):
    return _json_response({"ok": False, "error": message}, status=status)


def _audio_duration_seconds(path):
    try:
        info = sf.info(str(path))
        if info.samplerate:
            return round(info.frames / info.samplerate, 3)
    except Exception:
        return None
    return None


def _write_base64_audio(b64_data, out_path):
    """Decode base64 audio data and write to file. Supports data URI prefix."""
    b64_data = str(b64_data or "").strip()
    if b64_data.startswith("data:"):
        b64_data = b64_data.split(",", 1)[1] if "," in b64_data else b64_data
    audio_bytes = base64.b64decode(b64_data)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(audio_bytes)
    return out_path


def _read_audio_base64(path):
    """Read audio file and return base64 encoded string with data URI prefix."""
    data = Path(path).read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:audio/wav;base64,{b64}"


def _relative_path(path):
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(Path(path))


def _set_api_model(model, model_id, device):
    global _API_MODEL, _API_MODEL_ID, _API_DEVICE
    _API_MODEL = model
    _API_MODEL_ID = model_id
    _API_DEVICE = device


def _load_api_model_sync():
    logger.info(f"Loading model: {_API_MODEL_ID}, device: {_API_DEVICE} ...")
    from generator import load_miso_8b
    model = load_miso_8b(device=_API_DEVICE, model_path_or_repo_id=_API_MODEL_ID)
    logger.info("Model loaded!")
    return model


async def _ensure_api_model():
    global _API_MODEL
    if _API_MODEL is None:
        _API_MODEL = await asyncio.to_thread(_load_api_model_sync)
    return _API_MODEL


def _bool_option(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _normalize_seed(value):
    if value in (None, ""):
        return None
    try:
        return int(float(value)) % MISOTTS_SEED_MOD
    except (TypeError, ValueError):
        return None


def _sha256_text(value):
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _stable_seed_from_request(data, text, reference_audio_base64):
    explicit = _normalize_seed(data.get("seed") or data.get("misotts_seed"))
    if explicit is not None:
        return explicit
    if str(os.environ.get("MISOTTS_DETERMINISTIC", "1")).lower() in {"0", "false", "no", "off"}:
        return None
    payload = {
        "text": text,
        "reference_audio_sha256": _sha256_text(reference_audio_base64),
        "model_id": data.get("model_id") or _API_MODEL_ID,
        "temperature": data.get("temperature", 0.9),
        "topk": data.get("topk", 50),
        "max_audio_length_ms": data.get("max_audio_length_ms", 10000),
        "speaker": data.get("speaker", 0),
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return int(digest[:12], 16) % MISOTTS_SEED_MOD


def _apply_seed(seed):
    seed = _normalize_seed(seed)
    if seed is None:
        return None
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception:
        pass
    return seed


def _write_generated_audio(audio_tensor, sample_rate, out_path):
    waveform = audio_tensor.unsqueeze(0).cpu()
    torchaudio.save(str(out_path), waveform, sample_rate)


def _create_context_from_audio(audio_path, text, speaker):
    """Create context segments from reference audio."""
    from generator import Segment
    
    audio, sample_rate = torchaudio.load(str(audio_path))
    audio = audio.squeeze(0)
    
    # Resample if necessary
    target_sample_rate = 24000  # MisoTTS uses 24kHz
    if sample_rate != target_sample_rate:
        audio = torchaudio.functional.resample(
            audio, orig_freq=sample_rate, new_freq=target_sample_rate
        )
    
    return [Segment(speaker=speaker, text=text, audio=audio)]


def _synthesize_misotts_to_file(
    model,
    text,
    out_path,
    reference_audio=None,
    prompt_text="",
    speaker=0,
    temperature=0.9,
    topk=50,
    max_audio_length_ms=10000,
    seed=None,
):
    """Synthesize audio using MisoTTS model."""
    context = []
    if reference_audio:
        context = _create_context_from_audio(
            reference_audio, prompt_text, speaker
        )
    
    def generate_with_seed():
        _apply_seed(seed)
        return model.generate(
            text=text,
            speaker=speaker,
            context=context,
            max_audio_length_ms=max_audio_length_ms,
            temperature=temperature,
            topk=topk,
        )
    
    audio = generate_with_seed()
    _write_generated_audio(audio, model.sample_rate, out_path)


routes = web.RouteTableDef()


@routes.get("/api/health")
async def health(request):
    logger.info(f"[{request.method}] {request.path} from {request.remote}")
    return _json_response({"ok": True, "service": "misotts_api"})


@routes.get("/api/misotts/status")
async def status(request):
    logger.info(f"[{request.method}] {request.path} from {request.remote}")
    cached_models = []
    if _API_MODEL is not None:
        cached_models.append({
            "model_id": _API_MODEL_ID,
            "device": _API_DEVICE,
        })
    return _json_response({
        "ok": True,
        "models_cached": len(cached_models),
        "cached_models": cached_models,
    })


@routes.post("/api/misotts/unload")
async def unload(request):
    logger.info(f"[{request.method}] {request.path} from {request.remote}")
    global _API_MODEL
    count = 1 if _API_MODEL is not None else 0
    _API_MODEL = None
    import gc
    gc.collect()
    if sys.platform != "win32":
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    return _json_response({"ok": True, "unloaded": count})


@routes.post("/api/misotts/synthesize")
async def synthesize(request):
    client_ip = request.remote or "-"
    req_id = uuid.uuid4().hex[:8]
    logger.info(f"[{req_id}] [{request.method}] {request.path} from {client_ip}")

    try:
        data = await request.json()
    except Exception as exc:
        tb = traceback.format_exc()
        logger.warning(f"[{req_id}] Failed to parse JSON: {exc}\n{tb}")
        return _error(f"Invalid JSON body: {exc}\n{tb}", status=400)

    text = re.sub(r"\s+", " ", (data.get("text") or "").strip())
    if not text:
        logger.warning(f"[{req_id}] Missing text parameter")
        return _error("text is required and cannot be empty.")
    if len(text) > MAX_TEXT_LEN:
        logger.warning(f"[{req_id}] Text too long: {len(text)} > {MAX_TEXT_LEN}")
        return _error(f"text exceeds max length {MAX_TEXT_LEN}.")

    reference_audio_base64 = data.get("reference_audio_base64")
    prompt_text = re.sub(r"\s+", " ", (data.get("prompt_text") or "").strip())
    speaker = int(data.get("speaker", 0))
    temperature = float(data.get("temperature", 0.9))
    topk = int(data.get("topk", 50))
    max_audio_length_ms = int(data.get("max_audio_length_ms", 10000))
    seed = _stable_seed_from_request(data, text, reference_audio_base64)

    logger.info(
        f"[{req_id}] params: text_len={len(text)}, has_ref={bool(reference_audio_base64)}, "
        f"prompt_len={len(prompt_text)}, speaker={speaker}, "
        f"temperature={temperature}, topk={topk}, "
        f"max_audio_length_ms={max_audio_length_ms}, "
        f"seed={seed if seed is not None else '-'}"
    )

    out_dir = Path(data.get("output_dir") or OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = data.get("output_name")
    if not out_name:
        key = hashlib.sha256(
            json.dumps({
                "text": text,
                "prompt": prompt_text,
                "ref_b64_len": len(reference_audio_base64) if reference_audio_base64 else 0,
                "model": _API_MODEL_ID,
                "device": _API_DEVICE,
                "temperature": temperature,
                "topk": topk,
                "max_audio_length_ms": max_audio_length_ms,
                "speaker": speaker,
                "seed": seed,
            }, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:20]
        out_name = f"misotts_{key}.wav"
    out_path = out_dir / out_name

    ref_temp_path = None
    resolved_ref = None
    if reference_audio_base64:
        ref_temp_path = out_dir / f"ref_{uuid.uuid4().hex}.wav"
        try:
            _write_base64_audio(reference_audio_base64, ref_temp_path)
            resolved_ref = str(ref_temp_path)
            logger.info(f"[{req_id}] reference audio decoded: {ref_temp_path} ({ref_temp_path.stat().st_size} bytes)")
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error(f"[{req_id}] Failed to decode reference_audio_base64: {exc}\n{tb}")
            if ref_temp_path and ref_temp_path.exists():
                ref_temp_path.unlink(missing_ok=True)
            return _error(f"Failed to decode reference_audio_base64: {exc}\n{tb}")

    start_time = time.time()
    try:
        async with _API_LOCK:
            model = await _ensure_api_model()
            logger.info(f"[{req_id}] synthesis started -> {out_path}")
            await asyncio.to_thread(
                _synthesize_misotts_to_file,
                model,
                text,
                out_path,
                reference_audio=resolved_ref,
                prompt_text=prompt_text if resolved_ref else "",
                speaker=speaker,
                temperature=temperature,
                topk=topk,
                max_audio_length_ms=max_audio_length_ms,
                seed=seed,
            )
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error(f"[{req_id}] Synthesis failed: {exc}\n{tb}")
        if ref_temp_path and ref_temp_path.exists():
            ref_temp_path.unlink(missing_ok=True)
        return _error(f"Synthesis failed: {exc}\n{tb}", status=502)

    elapsed = round(time.time() - start_time, 3)
    if not out_path.exists():
        logger.error(f"[{req_id}] Output file not created: {out_path}")
        if ref_temp_path and ref_temp_path.exists():
            ref_temp_path.unlink(missing_ok=True)
        return _error("Synthesis finished but output file was not created.", status=502)

    audio_duration = _audio_duration_seconds(out_path)
    logger.info(
        f"[{req_id}] synthesis finished in {elapsed}s, output: {out_path} "
        f"({out_path.stat().st_size} bytes), audio_duration={audio_duration}"
    )

    try:
        output_base64 = _read_audio_base64(out_path)
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error(f"[{req_id}] Failed to encode output audio: {exc}\n{tb}")
        if ref_temp_path and ref_temp_path.exists():
            ref_temp_path.unlink(missing_ok=True)
        return _error(f"Failed to encode output audio: {exc}\n{tb}", status=502)

    if ref_temp_path and ref_temp_path.exists():
        ref_temp_path.unlink(missing_ok=True)

    logger.info(f"[{req_id}] response sent, audio_base64_len={len(output_base64)}")
    return _json_response({
        "ok": True,
        "audio_base64": output_base64,
        "output_path": str(out_path.resolve()),
        "relative_path": _relative_path(out_path),
        "elapsed_seconds": elapsed,
        "audio_duration_seconds": audio_duration,
        "seed": seed,
    })


@routes.get("/")
async def index(request):
    return web.Response(
        content_type="text/html",
        text="""<!doctype html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>MisoTTS API</title></head>
<body>
  <h1>MisoTTS API Server</h1>
  <pre>
GET  /api/health
GET  /api/misotts/status
POST /api/misotts/unload
POST /api/misotts/synthesize

Request (JSON):
{
  "text": "要合成的文本",
  "reference_audio_base64": "data:audio/wav;base64,xxxx...",  // 可选
  "prompt_text": "参考音频对应的文本",                         // 可选；仅随 reference_audio 使用
  "model_id": "MisoLabs/MisoTTS",
  "device": "auto",
  "speaker": 0,
  "temperature": 0.9,
  "topk": 50,
  "max_audio_length_ms": 10000,
  "seed": 123456789  // 可选；不传时默认按文本、参考音频和生成参数派生稳定 seed
}

Response (JSON):
{
  "ok": true,
  "audio_base64": "data:audio/wav;base64,xxxx...",
  "output_path": "/abs/path/to/output.wav",
  "relative_path": "work/misotts_api_outputs/misotts_xxx.wav",
  "elapsed_seconds": 12.345,
  "audio_duration_seconds": 2.431,
  "seed": 123456789
}
  </pre>
</body>
</html>""",
    )


async def on_startup(app):
    print(f"[MisoTTS API] listening on http://{app['host']}:{app['port']} (max request {MAX_REQUEST_MB} MB)")


def main(argv=None):
    parser = argparse.ArgumentParser(description="MisoTTS API")
    parser.add_argument("--model", default="MisoLabs/MisoTTS", help="Model path or HuggingFace repo ID")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu)")
    parser.add_argument("--ip", default="localhost", help="Server IP")
    parser.add_argument("--port", type=int, default=6006, help="Server port")
    args = parser.parse_args(argv)

    device = args.device or get_best_device()
    _set_api_model(None, args.model, device)

    app = web.Application(client_max_size=MAX_REQUEST_SIZE)
    app["host"] = args.ip
    app["port"] = args.port
    app.add_routes(routes)
    app.on_startup.append(on_startup)
    web.run_app(app, host=args.ip, port=args.port)


if __name__ == "__main__":
    raise SystemExit(main())