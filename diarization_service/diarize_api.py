import base64
import tempfile
from typing import Optional
import os

import torch
import pandas as pd
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uvicorn
from pyannote.audio import Pipeline
from logger_utils import logger, logger_middleware
from utils import load_audio
import warnings
warnings.filterwarnings("ignore")


# ---------------------------
# Pydantic Request Model
# ---------------------------
class DiarizationRequest(BaseModel):
    audio_base64: str  # base64 encoded audio file
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    req_id: str


# ---------------------------
# Lifespan Context Manager
# ---------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading Pyannote diarization pipeline...")

    app.state.diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=os.getenv("HF_TOKEN"),
        cache_dir="./models",
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    logger.info("Pyannote diarization model loaded successfully.")
    yield
    logger.info("Shutting down diarization service...")


# ---------------------------
# FastAPI App
# ---------------------------
app = FastAPI(title="Diarization Service", version="1.0.0", lifespan=lifespan)
app.middleware("http")(logger_middleware)


# ---------------------------
# Diarization Pipeline Logic
# ---------------------------
def run_diarization(
    pipeline: Pipeline,
    audio_path: Path,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> pd.DataFrame:

    audio = load_audio(str(audio_path))
    audio_data = {"waveform": torch.from_numpy(audio[None, :]), "sample_rate": 16000}

    diarization = pipeline(
        audio_data, min_speakers=min_speakers, max_speakers=max_speakers
    )

    diarize_df = pd.DataFrame(
        diarization.itertracks(yield_label=True),
        columns=["segment", "label", "speaker"],
    )
    diarize_df["start"] = diarize_df["segment"].apply(lambda x: x.start)
    diarize_df["end"] = diarize_df["segment"].apply(lambda x: x.end)

    return diarize_df


# ---------------------------
# API Endpoint
# ---------------------------
@app.post("/diarize")
async def diarize(payload: DiarizationRequest):
    try:
        pipeline: Pipeline = app.state.diarization_pipeline

        # Decode base64 → bytes
        audio_bytes = base64.b64decode(payload.audio_base64)

        # Save to a temporary WAV file
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        with open(temp_input.name, "wb") as f:
            f.write(audio_bytes)

        diarize_df = run_diarization(
            pipeline=pipeline,
            audio_path=Path(temp_input.name),
            min_speakers=payload.min_speakers,
            max_speakers=payload.max_speakers,
        )

        result = diarize_df.to_dict(orient="records")
        logger.info(f"Diarization complete: {len(result)} segments detected")
        return {"status": "success", "data": result}

    except Exception as e:
        logger.error(f"Error during diarization: {e}")
        return {"status": "error", "message": str(e)}


# ---------------------------
# Run with uvicorn internally
# ---------------------------
if __name__ == "__main__":
    uvicorn.run(
        "diarize_api:app",
        host="0.0.0.0",
        port=4504,
    )
