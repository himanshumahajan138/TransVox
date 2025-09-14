import base64
import uvicorn
from pydantic import BaseModel
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException

from utils import IndicASRService, LANGUAGES
from logger_utils import logger, logger_middleware


# ---------------------------
# Pydantic Request Model
# ---------------------------
class TranscriptionRequest(BaseModel):
    language: str  # full language name (e.g., "hindi", "malayalam")
    audio_base64: str
    req_id: str


# ---------------------------
# Lifespan Context
# ---------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Indic ASR API service...")
    app.state.asr_service = IndicASRService()  # Load once at startup
    yield
    logger.info("Shutting down Indic ASR API service...")


# ---------------------------
# FastAPI App
# ---------------------------
app = FastAPI(title="Indic ASR API", version="2.1.0", lifespan=lifespan)
app.middleware("http")(logger_middleware)


# ---------------------------
# Endpoint
# ---------------------------
@app.post("/transcribe/")
async def transcribe_audio(req: TranscriptionRequest):
    """
    Transcribe audio using Base64 encoded input.
    - language: Full language name (e.g., 'hindi', 'malayalam')
    - audio_base64: Base64 encoded audio
    - req_id: Unique request identifier
    """
    try:
        language = str(req.language).lower()
        logger.info(f"Received transcription request for language={language}")

        # Validate language
        if language not in LANGUAGES:
            logger.error(f"Unsupported language: {language}")
            raise HTTPException(status_code=400, detail="Unsupported language")

        # Decode base64 → bytes
        try:
            audio_bytes = base64.b64decode(req.audio_base64)
        except Exception as e:
            logger.error(f"[{req.req_id}] Base64 decoding failed: {e}")
            raise HTTPException(status_code=400, detail="Invalid base64 audio input")

        # Run transcription
        transcription = app.state.asr_service.transcribe(audio_bytes, language)
        logger.info("Transcription completed successfully")
        return JSONResponse(
            content={
                "status": "success",
                "req_id": req.req_id,
                "language_name": language,
                "language_code": LANGUAGES[language],
                "data": transcription,
            }
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ---------------------------
# Run with Uvicorn
# ---------------------------
if __name__ == "__main__":
    uvicorn.run("indic_conformer_api:app", host="0.0.0.0", port=6004, reload=False)
