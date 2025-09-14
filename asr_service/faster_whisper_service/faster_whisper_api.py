import tempfile
from pathlib import Path
from contextlib import asynccontextmanager
from pydantic import BaseModel
import torch
import uvicorn
import base64
from fastapi import FastAPI, HTTPException
from logger_utils import logger, logger_middleware
from faster_whisper import WhisperModel

# -------------------------
# Language Map + Params
# -------------------------
# fmt: off
LANGUAGES = {'english': 'en', 'chinese': 'zh', 'german': 'de', 'spanish': 'es', 'russian': 'ru', 'korean': 'ko', 'french': 'fr', 'japanese': 'ja', 'portuguese': 'pt', 'turkish': 'tr', 'polish': 'pl', 'catalan': 'ca', 'dutch': 'nl', 'arabic': 'ar', 'swedish': 'sv', 'italian': 'it', 'indonesian': 'id', 'hindi': 'hi', 'finnish': 'fi', 'vietnamese': 'vi', 'hebrew': 'he', 'ukrainian': 'uk', 'greek': 'el', 'malay': 'ms', 'czech': 'cs', 'romanian': 'ro', 'danish': 'da', 'hungarian': 'hu', 'tamil': 'ta', 'norwegian': 'no', 'thai': 'th', 'urdu': 'ur', 'croatian': 'hr', 'bulgarian': 'bg', 'lithuanian': 'lt', 'latin': 'la', 'maori': 'mi', 'malayalam': 'ml', 'welsh': 'cy', 'slovak': 'sk', 'telugu': 'te', 'persian': 'fa', 'latvian': 'lv', 'bengali': 'bn', 'serbian': 'sr', 'azerbaijani': 'az', 'slovenian': 'sl', 'kannada': 'kn', 'estonian': 'et', 'macedonian': 'mk', 'breton': 'br', 'basque': 'eu', 'icelandic': 'is', 'armenian': 'hy', 'nepali': 'ne', 'mongolian': 'mn', 'bosnian': 'bs', 'kazakh': 'kk', 'albanian': 'sq', 'swahili': 'sw', 'galician': 'gl', 'marathi': 'mr', 'punjabi': 'pa', 'sinhala': 'si', 'khmer': 'km', 'shona': 'sn', 'yoruba': 'yo', 'somali': 'so', 'afrikaans': 'af', 'occitan': 'oc', 'georgian': 'ka', 'belarusian': 'be', 'tajik': 'tg', 'sindhi': 'sd', 'gujarati': 'gu', 'amharic': 'am', 'yiddish': 'yi', 'lao': 'lo', 'uzbek': 'uz', 'faroese': 'fo', 'haitian creole': 'ht', 'pashto': 'ps', 'turkmen': 'tk', 'nynorsk': 'nn', 'maltese': 'mt', 'sanskrit': 'sa', 'luxembourgish': 'lb', 'myanmar': 'my', 'tibetan': 'bo', 'tagalog': 'tl', 'malagasy': 'mg', 'assamese': 'as', 'tatar': 'tt', 'hawaiian': 'haw', 'lingala': 'ln', 'hausa': 'ha', 'bashkir': 'ba', 'javanese': 'jw', 'sundanese': 'su', 'cantonese': 'yue', 'burmese': 'my', 'valencian': 'ca', 'flemish': 'nl', 'haitian': 'ht', 'letzeburgesch': 'lb', 'pushto': 'ps', 'panjabi': 'pa', 'moldavian': 'ro', 'moldovan': 'ro', 'sinhalese': 'si', 'castilian': 'es', 'mandarin': 'zh'}
FASTER_WHISPER_PARAMS = {
    "task": "transcribe",
    "log_progress": False,
    "beam_size": 5,
    "best_of": 5,
    "patience": 1,
    "length_penalty": 1,
    "repetition_penalty": 1,
    "no_repeat_ngram_size": 0,
    "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "compression_ratio_threshold": 2.4,
    "log_prob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    "condition_on_previous_text": True,
    "prompt_reset_on_temperature": 0.5,
    "initial_prompt": None,
    "prefix": None,
    "suppress_blank": True,
    "suppress_tokens": [-1],
    "without_timestamps": False,
    "max_initial_timestamp": 1.0,
    "prepend_punctuations": "\"'“¿([{-",
    "append_punctuations": "\"'.。,，!！?？:：”)]}、",
    "multilingual": False,
    "vad_filter": False,
    "vad_parameters": None,
    "max_new_tokens": None,
    "chunk_length": None,
    "clip_timestamps": "0",
    "hallucination_silence_threshold": None,
    "hotwords": None,
    "language_detection_threshold": 0.5,
    "language_detection_segments": 1,
}
# fmt: on


class TranscribeRequest(BaseModel):
    audio_base64: str  # base64-encoded audio
    language: str = "english"
    wlt: bool = False
    req_id: str


# -------------------------
# Lifespan: Model Management
# -------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading Faster Whisper model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WhisperModel("large-v3", device=device, compute_type="float16", download_root="./models/")
    app.state.faster_whisper_model = model
    logger.info(f"Faster Whisper model loaded on {device}.")
    yield
    logger.info("Shutting down Faster Whisper service... cleanup done.")


app = FastAPI(title="Faster Whisper Service", version="1.0.0", lifespan=lifespan)
app.middleware("http")(logger_middleware)


# -------------------------
# Faster Whisper Pipeline
# -------------------------
def faster_whisper_pipeline(audio_path: Path, model: WhisperModel, language: str, wlt: bool):
    if not audio_path.exists():
        raise FileNotFoundError("Provided audio path does not exist.")

    segments, _ = model.transcribe(
        audio=audio_path,
        word_timestamps=wlt,
        language=LANGUAGES.get(language),
        **FASTER_WHISPER_PARAMS,
    )
    return list(segments)


# -------------------------
# API Endpoint
# -------------------------
@app.post("/transcribe")
async def transcribe_audio(payload: TranscribeRequest):
    try:
        logger.info(
            f"Received transcription request: lang={payload.language}, word_level_timestamps={payload.wlt}"
        )

        # Decode base64 → bytes
        audio_bytes = base64.b64decode(payload.audio_base64)

        # Save to a temporary WAV file
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        with open(temp_input.name, "wb") as f:
            f.write(audio_bytes)

        # Validate language
        if payload.language.lower() not in LANGUAGES:
            raise HTTPException(
                status_code=400, detail=f"Unsupported language: {payload.language}"
            )

        # Run Whisper pipeline
        segments = faster_whisper_pipeline(
            audio_path=Path(temp_input.name),
            model=app.state.faster_whisper_model,
            language=payload.language.lower(),
            wlt=payload.wlt,
        )

        logger.info("Returning transcription result.")
        return {"status": "success", "data": segments}

    except Exception as e:
        logger.exception("Error in Whisper transcription pipeline")
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------
# Run with Uvicorn Internally
# -------------------------
if __name__ == "__main__":
    uvicorn.run(
        "faster_whisper_api:app",
        host="0.0.0.0",
        port=6003,
    )
