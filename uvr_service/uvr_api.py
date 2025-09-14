import uuid
import base64
import tempfile
from pathlib import Path
from pydantic import BaseModel
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from audio_separator.separator import Separator
from logger_utils import logger, logger_middleware


class UVRRequest(BaseModel):
    audio_base64: str  # base64-encoded audio
    req_id: str


# -------------------------
# Lifespan: Model Management
# -------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading UVR model into memory...")
    model = Separator(
        model_file_dir="./models", output_dir=tempfile.gettempdir(), sample_rate=16000
    )
    model.load_model(model_filename="UVR-MDX-NET-Inst_HQ_5.onnx")
    app.state.uvr_model = model
    logger.info("UVR model loaded successfully.")
    yield
    logger.info("Shutting down UVR service... cleanup done.")


app = FastAPI(title="UVR Service", version="1.0.0", lifespan=lifespan)
app.middleware("http")(logger_middleware)


# -------------------------
# Core UVR Pipeline
# -------------------------
def uvr_pipeline(
    audio_path: Path,
    model,
) -> str:
    if not audio_path.exists():
        raise FileNotFoundError("Provided audio path does not exist.")

    output_files = model.separate([audio_path])
    vocal_file,instrumental_file = "", ""
    temp_dir = tempfile.gettempdir()
    for file in output_files:
        if "(Vocals)" in file:
            vocal_file = temp_dir + "/" + file
        if "(Instrumental)" in file:
            instrumental_file = temp_dir + "/" + file

    return vocal_file, instrumental_file


# -------------------------
# API Endpoints
# -------------------------
@app.post("/uvr")
async def run_vad(payload: UVRRequest):
    try:
        logger.info("Received UVR request")

        # Decode base64 → bytes
        audio_bytes = base64.b64decode(payload.audio_base64)

        # Save to a temporary WAV file
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        with open(temp_input.name, "wb") as f:
            f.write(audio_bytes)

        # Run pipeline
        vocal_file, instrumental_file = uvr_pipeline(
            audio_path=Path(temp_input.name),
            model=app.state.uvr_model,
        )
        
        if not vocal_file or not instrumental_file:
            logger.exception("Either Vocal or Instrumental File Don't exists in UVR pipeline")
            raise HTTPException(status_code=500, detail=str(e))

        # Read output file and encode to base64
        with open(vocal_file, "rb") as v, open(instrumental_file, "rb") as i:
            vocal_base64 = base64.b64encode(v.read()).decode("utf-8")
            instrumental_base64 = base64.b64encode(i.read()).decode("utf-8")

        logger.info("Returning vocal and instrumental separated audios.")
        return JSONResponse(
            {
                "status": "success",
                "vocal_base64": vocal_base64,
                "instrumental_base64": instrumental_base64,
            }
        )

    except Exception as e:
        logger.exception("Error in UVR pipeline")
        raise HTTPException(status_code=500, detail=str(e))


# ----------
# Run
# ----------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "uvr_api:app",
        host="0.0.0.0",
        port=6000,
    )
