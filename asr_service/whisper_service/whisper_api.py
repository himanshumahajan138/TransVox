import tempfile
from pathlib import Path
from contextlib import asynccontextmanager
from pydantic import BaseModel
import torch
import uvicorn
import base64
from fastapi import FastAPI, HTTPException
from logger_utils import logger, logger_middleware
from whisper import load_model, load_audio, transcribe

# -------------------------
# Language Map + Params
# -------------------------
# fmt: off
LANGUAGES = {'english': 'en', 'chinese': 'zh', 'german': 'de', 'spanish': 'es', 'russian': 'ru', 'korean': 'ko', 'french': 'fr', 'japanese': 'ja', 'portuguese': 'pt', 'turkish': 'tr', 'polish': 'pl', 'catalan': 'ca', 'dutch': 'nl', 'arabic': 'ar', 'swedish': 'sv', 'italian': 'it', 'indonesian': 'id', 'hindi': 'hi', 'finnish': 'fi', 'vietnamese': 'vi', 'hebrew': 'he', 'ukrainian': 'uk', 'greek': 'el', 'malay': 'ms', 'czech': 'cs', 'romanian': 'ro', 'danish': 'da', 'hungarian': 'hu', 'tamil': 'ta', 'norwegian': 'no', 'thai': 'th', 'urdu': 'ur', 'croatian': 'hr', 'bulgarian': 'bg', 'lithuanian': 'lt', 'latin': 'la', 'maori': 'mi', 'malayalam': 'ml', 'welsh': 'cy', 'slovak': 'sk', 'telugu': 'te', 'persian': 'fa', 'latvian': 'lv', 'bengali': 'bn', 'serbian': 'sr', 'azerbaijani': 'az', 'slovenian': 'sl', 'kannada': 'kn', 'estonian': 'et', 'macedonian': 'mk', 'breton': 'br', 'basque': 'eu', 'icelandic': 'is', 'armenian': 'hy', 'nepali': 'ne', 'mongolian': 'mn', 'bosnian': 'bs', 'kazakh': 'kk', 'albanian': 'sq', 'swahili': 'sw', 'galician': 'gl', 'marathi': 'mr', 'punjabi': 'pa', 'sinhala': 'si', 'khmer': 'km', 'shona': 'sn', 'yoruba': 'yo', 'somali': 'so', 'afrikaans': 'af', 'occitan': 'oc', 'georgian': 'ka', 'belarusian': 'be', 'tajik': 'tg', 'sindhi': 'sd', 'gujarati': 'gu', 'amharic': 'am', 'yiddish': 'yi', 'lao': 'lo', 'uzbek': 'uz', 'faroese': 'fo', 'haitian creole': 'ht', 'pashto': 'ps', 'turkmen': 'tk', 'nynorsk': 'nn', 'maltese': 'mt', 'sanskrit': 'sa', 'luxembourgish': 'lb', 'myanmar': 'my', 'tibetan': 'bo', 'tagalog': 'tl', 'malagasy': 'mg', 'assamese': 'as', 'tatar': 'tt', 'hawaiian': 'haw', 'lingala': 'ln', 'hausa': 'ha', 'bashkir': 'ba', 'javanese': 'jw', 'sundanese': 'su', 'cantonese': 'yue', 'burmese': 'my', 'valencian': 'ca', 'flemish': 'nl', 'haitian': 'ht', 'letzeburgesch': 'lb', 'pushto': 'ps', 'panjabi': 'pa', 'moldavian': 'ro', 'moldovan': 'ro', 'sinhalese': 'si', 'castilian': 'es', 'mandarin': 'zh'}
WHISPER_PARAMS = {
    "verbose": None,
    "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    "compression_ratio_threshold": 2.4,
    "logprob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    "condition_on_previous_text": True,
    "initial_prompt": None,
    "carry_initial_prompt": False,
    "prepend_punctuations": "\"'“¿([{-",
    "append_punctuations": "\"'.。,，!！?？:：”)]}、",
    "clip_timestamps": "0",
    "hallucination_silence_threshold": None,
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
    logger.info("Loading Whisper model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("large-v3", device=device, download_root="./models/")
    app.state.whisper_model = model
    logger.info(f"Whisper model loaded on {device}.")
    yield
    logger.info("Shutting down Whisper service... cleanup done.")


app = FastAPI(title="Whisper Service", version="1.0.0", lifespan=lifespan)
app.middleware("http")(logger_middleware)


# -------------------------
# Whisper Pipeline
# -------------------------
def whisper_pipeline(audio_path: Path, model, language: str, wlt: bool):
    if not audio_path.exists():
        raise FileNotFoundError("Provided audio path does not exist.")

    wav = load_audio(str(audio_path))
    result = transcribe(
        model=model,
        audio=wav,
        word_timestamps=wlt,
        language=LANGUAGES.get(language),
        **WHISPER_PARAMS,
    )
    return result["segments"]


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
        result = whisper_pipeline(
            audio_path=Path(temp_input.name),
            model=app.state.whisper_model,
            language=payload.language.lower(),
            wlt=payload.wlt,
        )

        logger.info("Returning transcription result.")
        return {"status": "success", "data": result}

    except Exception as e:
        logger.exception("Error in Whisper transcription pipeline")
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------
# Run with Uvicorn Internally
# -------------------------
if __name__ == "__main__":
    uvicorn.run(
        "whisper_api:app",
        host="0.0.0.0",
        port=6005,
    )
