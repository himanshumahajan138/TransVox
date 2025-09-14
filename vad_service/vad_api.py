import tempfile
from pathlib import Path
from logger_utils import logger, logger_middleware
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from silero_vad import (
    load_silero_vad,
    read_audio,
    get_speech_timestamps,
    save_audio,
    collect_chunks,
)
import base64
from pydantic import BaseModel


# -------------------------
# VAD Parameters
# -------------------------
VAD_PARAMS = {
    "threshold": 0.5,
    "min_speech_duration_ms": 250,
    "max_speech_duration_s": float("inf"),
    "min_silence_duration_ms": 100,
    "speech_pad_ms": 30,
    "return_seconds": False,
    "time_resolution": 1,
    "visualize_probs": False,
    "progress_tracking_callback": None,
    "neg_threshold": None,
    "window_size_samples": 512,
    "min_silence_at_max_speech": 98,
    "use_max_poss_sil_at_max_speech": True,
}


class VADRequest(BaseModel):
    audio_base64: str  # base64-encoded audio
    sample_rate: int = 16000
    req_id: str


# -------------------------
# Lifespan: Model Management
# -------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading VAD model into memory...")
    model = load_silero_vad()
    app.state.vad_model = model
    logger.info("VAD model loaded successfully.")
    yield
    logger.info("Shutting down VAD service... cleanup done.")


app = FastAPI(title="VAD Service", version="1.0.0", lifespan=lifespan)
app.middleware("http")(logger_middleware)


# -------------------------
# Core VAD Pipeline
# -------------------------
def vad_pipeline(
    audio_path: Path,
    model,
    sample_rate: int = 16000,
) -> str:
    if not audio_path.exists():
        raise FileNotFoundError("Provided audio path does not exist.")

    wav = read_audio(audio_path, sample_rate)
    speech_timestamps = get_speech_timestamps(
        audio=wav, model=model, sampling_rate=sample_rate, **VAD_PARAMS
    )

    vad_processed_audio = collect_chunks(
        tss=speech_timestamps,
        wav=wav,
        seconds=VAD_PARAMS.get("return_seconds", False),
        sampling_rate=sample_rate,
    )

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    save_audio(temp_file.name, vad_processed_audio, sample_rate)
    return speech_timestamps, temp_file.name


# -------------------------
# API Endpoints
# -------------------------
@app.post("/vad")
async def run_vad(payload: VADRequest):
    try:
        logger.info(f"Received VAD request: sample_rate={payload.sample_rate}")

        # Decode base64 → bytes
        audio_bytes = base64.b64decode(payload.audio_base64)

        # Save to a temporary WAV file
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        with open(temp_input.name, "wb") as f:
            f.write(audio_bytes)

        # Run pipeline
        speech_timestamps, output_file = vad_pipeline(
            audio_path=Path(temp_input.name),
            model=app.state.vad_model,
            sample_rate=payload.sample_rate,
        )

        # Read output file and encode to base64
        with open(output_file, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode("utf-8")

        logger.info("Returning speech timestamps + processed audio (base64).")
        return JSONResponse(
            {
                "status": "success",
                "speech_timestamps": speech_timestamps,
                "audio_base64": audio_base64,
            }
        )

    except Exception as e:
        logger.exception("Error in VAD pipeline")
        raise HTTPException(status_code=500, detail=str(e))


# ----------
# Run
# ----------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "vad_api:app",
        host="0.0.0.0",
        port=6001,
    )
