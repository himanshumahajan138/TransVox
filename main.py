import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from utils import stt_pipeline
from logger_utils import logger, logger_middleware


# ---------------------------
# Pydantic Model
# ---------------------------
class PipelineRequest(BaseModel):
    audio_url: str = ""
    stt_service: str
    start_pattern: str = "[{"
    end_pattern: str = "}]: "
    language: str = "english"
    wlt: bool = True
    uvr: bool = True
    vad: bool = True
    diarize: bool = False
    output_format: str
    req_id: str
    audio_path: str = ""
    maintain_gaps: bool = True


# ---------------------------
# App Initialization
# ---------------------------
app = FastAPI(title="Transcription Pipeline", version="1.0.0")
app.middleware("http")(logger_middleware)


# ---------------------------
# Endpoint
# ---------------------------
@app.post("/speech-to-text-service")
async def process_pipeline(payload: PipelineRequest):
    try:
        logger.info(
            f"Received request | req_id={payload.req_id}, "
            f"stt_service={payload.stt_service}, "
            f"audio_url={payload.audio_url}, "
            f"language={payload.language}, "
            f"wlt={payload.wlt}, "
            f"uvr={payload.uvr}, "
            f"vad={payload.vad}, "
            f"diarize={payload.diarize}, "
            f"audio_path={payload.audio_path}, "
            f"output_format={payload.output_format}, "
            f"maintain_gaps={payload.maintain_gaps}"
        )

        result = stt_pipeline(
            stt_service=payload.stt_service,
            start_pattern=payload.start_pattern,
            end_pattern=payload.end_pattern,
            language=payload.language,
            wlt=payload.wlt,
            uvr=payload.uvr,
            vad=payload.vad,
            diarize=payload.diarize,
            req_id=payload.req_id,
            output_format=payload.output_format,
            audio_url=payload.audio_url,
            audio_path=payload.audio_path,
            maintain_gaps=payload.maintain_gaps,
        )

        logger.info(f"Completed processing | req_id={payload.req_id}")
        if result.get("status") == "success":
            return {
                "status": "success",
                "req_id": payload.req_id,
                "output_file_path": result.get("output_file"),
                "transcript": result.get("transcript"),
                "segments": result.get("segments"),
                "original_speakers": result.get("original_speakers"),
            }
        else:
            error_message = result.get("message", "Transcription failed.")
            logger.error(f"Pipeline error | req_id={payload.req_id} | {error_message}")
            raise HTTPException(status_code=422, detail=error_message)

    except ValueError as ve:
        logger.error(f"ValueError | req_id={payload.req_id} | {ve}")
        raise HTTPException(status_code=400, detail=str(ve))

    except HTTPException as he:
        logger.error(f"HTTPException | req_id={payload.req_id} | {he.detail}")
        raise he

    except Exception as e:
        logger.exception(f"Unexpected error | req_id={payload.req_id} | {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error. Please check logs for more details.",
        )


# ---------------------------
# Run server
# ---------------------------
if __name__ == "__main__":
    # uvicorn.run("main:app", host="0.0.0.0", port=4505)
    uvicorn.run("main:app", host="0.0.0.0", port=9001)
