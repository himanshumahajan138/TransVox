import io
import torch
import torchaudio
from logger_utils import logger
import nemo.collections.asr as nemo_asr


# ----------------- Language Mapping -----------------
LANGUAGES = {
    "assamese": "as",
    "bengali": "bn",
    "bodo": "br",
    "dogri": "doi",
    "gujarati": "gu",
    "hindi": "hi",
    "kannada": "kn",
    "kashmiri": "ks",
    "konkani": "kok",
    "maithili": "mai",
    "malayalam": "ml",
    "manipuri": "mni",
    "marathi": "mr",
    "nepali": "ne",
    "odia": "or",
    "punjabi": "pa",
    "sanskrit": "sa",
    "santali": "sat",
    "sindhi": "sd",
    "tamil": "ta",
    "telugu": "te",
    "urdu": "ur",
}


# ----------------- ASR Service -----------------
class IndicASRService:
    def __init__(self, model_name="ai4bharat/IndicConformer"):
        logger.info(f"Loading ASR model: {model_name}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = nemo_asr.models.EncDecCTCModel.from_pretrained(
            model_name=model_name
        ).to(device)
        self.model.eval()
        logger.info(f"Model loaded successfully on {device}")

    def transcribe(self, audio_bytes: bytes, language_name: str) -> str:
        try:
            # validate & map language
            if language_name not in LANGUAGES:
                raise ValueError(f"Unsupported language: {language_name}")
            lang_code = LANGUAGES[language_name]

            signal, sr = torchaudio.load(io.BytesIO(audio_bytes))
            logger.debug(f"Original sample rate: {sr}, shape: {signal.shape}")

            if signal.shape[0] > 1:  # stereo → mono
                signal = torch.mean(signal, dim=0, keepdim=True)

            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            signal = resampler(signal).squeeze().numpy()

            logger.info(
                f"Running transcription | Language={language_name} ({lang_code})"
            )
            results = self.model.transcribe(
                [signal], batch_size=1, logprobs=False, language_id=lang_code
            )
            return results[0][0]
        except Exception as e:
            logger.exception("Error during transcription")
            raise e
