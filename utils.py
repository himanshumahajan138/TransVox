import os
import re
import uuid
import pysrt
import bisect
import base64
import requests
import tempfile
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from logger_utils import logger
from typing import List, Optional
from pydantic import Field, BaseModel
from typing import Union, Dict, Any, List

# ---------------------------
# External Microservice URLs
# ---------------------------
UVR_URL = "http://0.0.0.0:6000/uvr"
VAD_URL = "http://0.0.0.0:6001/vad"
WHISPER_URL = "http://0.0.0.0:6002/transcribe"
FASTER_WHISPER_URL = "http://0.0.0.0:6003/transcribe"
INDIC_CONFORMER_URL = "http://0.0.0.0:6004/transcribe"
DIARIZE_URL = "http://0.0.0.0:6005/diarize"

INDIC_CONFORMER_LANGUAGES = {
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
# fmt:off
WHISPER_LANGUAGES = {'english': 'en', 'chinese': 'zh', 'german': 'de', 'spanish': 'es', 'russian': 'ru', 'korean': 'ko', 'french': 'fr', 'japanese': 'ja', 'portuguese': 'pt', 'turkish': 'tr', 'polish': 'pl', 'catalan': 'ca', 'dutch': 'nl', 'arabic': 'ar', 'swedish': 'sv', 'italian': 'it', 'indonesian': 'id', 'hindi': 'hi', 'finnish': 'fi', 'vietnamese': 'vi', 'hebrew': 'he', 'ukrainian': 'uk', 'greek': 'el', 'malay': 'ms', 'czech': 'cs', 'romanian': 'ro', 'danish': 'da', 'hungarian': 'hu', 'tamil': 'ta', 'norwegian': 'no', 'thai': 'th', 'urdu': 'ur', 'croatian': 'hr', 'bulgarian': 'bg', 'lithuanian': 'lt', 'latin': 'la', 'maori': 'mi', 'malayalam': 'ml', 'welsh': 'cy', 'slovak': 'sk', 'telugu': 'te', 'persian': 'fa', 'latvian': 'lv', 'bengali': 'bn', 'serbian': 'sr', 'azerbaijani': 'az', 'slovenian': 'sl', 'kannada': 'kn', 'estonian': 'et', 'macedonian': 'mk', 'breton': 'br', 'basque': 'eu', 'icelandic': 'is', 'armenian': 'hy', 'nepali': 'ne', 'mongolian': 'mn', 'bosnian': 'bs', 'kazakh': 'kk', 'albanian': 'sq', 'swahili': 'sw', 'galician': 'gl', 'marathi': 'mr', 'punjabi': 'pa', 'sinhala': 'si', 'khmer': 'km', 'shona': 'sn', 'yoruba': 'yo', 'somali': 'so', 'afrikaans': 'af', 'occitan': 'oc', 'georgian': 'ka', 'belarusian': 'be', 'tajik': 'tg', 'sindhi': 'sd', 'gujarati': 'gu', 'amharic': 'am', 'yiddish': 'yi', 'lao': 'lo', 'uzbek': 'uz', 'faroese': 'fo', 'haitian creole': 'ht', 'pashto': 'ps', 'turkmen': 'tk', 'nynorsk': 'nn', 'maltese': 'mt', 'sanskrit': 'sa', 'luxembourgish': 'lb', 'myanmar': 'my', 'tibetan': 'bo', 'tagalog': 'tl', 'malagasy': 'mg', 'assamese': 'as', 'tatar': 'tt', 'hawaiian': 'haw', 'lingala': 'ln', 'hausa': 'ha', 'bashkir': 'ba', 'javanese': 'jw', 'sundanese': 'su', 'cantonese': 'yue', 'burmese': 'my', 'valencian': 'ca', 'flemish': 'nl', 'haitian': 'ht', 'letzeburgesch': 'lb', 'pushto': 'ps', 'panjabi': 'pa', 'moldavian': 'ro', 'moldovan': 'ro', 'sinhalese': 'si', 'castilian': 'es', 'mandarin': 'zh'}
# fmt:on

class Word(BaseModel):
    start: Optional[float] = Field(default=None, description="Start time of the word")
    end: Optional[float] = Field(default=None, description="Start time of the word")
    word: Optional[str] = Field(default=None, description="Word text")
    probability: Optional[float] = Field(
        default=None, description="Probability of the word"
    )


class Segment(BaseModel):
    id: Optional[int] = Field(
        default=None, description="Incremental id for the segment"
    )
    seek: Optional[int] = Field(
        default=None, description="Seek of the segment from chunked audio"
    )
    text: Optional[str] = Field(
        default=None, description="Transcription text of the segment"
    )
    start: Optional[float] = Field(
        default=None, description="Start time of the segment"
    )
    end: Optional[float] = Field(default=None, description="End time of the segment")
    tokens: Optional[List[int]] = Field(default=None, description="List of token IDs")
    temperature: Optional[float] = Field(
        default=None, description="Temperature used during the decoding process"
    )
    avg_logprob: Optional[float] = Field(
        default=None, description="Average log probability of the tokens"
    )
    compression_ratio: Optional[float] = Field(
        default=None, description="Compression ratio of the segment"
    )
    no_speech_prob: Optional[float] = Field(
        default=None, description="Probability that it's not speech"
    )
    words: Optional[List["Word"]] = Field(
        default=None, description="List of words contained in the segment"
    )


class SpeechTimestampsMap:
    """Helper class to restore original speech timestamps."""

    def __init__(
        self,
        chunks: List[dict],
        sampling_rate: int,
        time_precision: int = 2,
        maintain_gaps: bool = True,
    ):
        self.sampling_rate = sampling_rate
        self.time_precision = time_precision
        self.maintain_gaps = maintain_gaps
        self.chunk_end_sample = []
        self.total_silence_before = []

        previous_end = 0
        silent_samples = 0

        for chunk in chunks:
            silent_samples += chunk["start"] - previous_end
            previous_end = chunk["end"]
            
            if maintain_gaps:
                self.chunk_end_sample.append(chunk["end"])
            else:
                self.chunk_end_sample.append(chunk["end"] - silent_samples)
            self.total_silence_before.append(silent_samples / sampling_rate)

    def get_original_time(
        self,
        time: float,
        chunk_index: Optional[int] = None,
        is_end: bool = False,
    ) -> float:
        if chunk_index is None:
            chunk_index = self.get_chunk_index(time, is_end)

        total_silence_before = (
            0 if self.maintain_gaps else self.total_silence_before[chunk_index]
        )
        return round(total_silence_before + time, self.time_precision)

    def get_chunk_index(self, time: float, is_end: bool = False) -> int:
        sample = int(time * self.sampling_rate)
        if sample in self.chunk_end_sample and is_end:
            return self.chunk_end_sample.index(sample)

        return min(
            bisect.bisect(self.chunk_end_sample, sample),
            len(self.chunk_end_sample) - 1,
        )


def format_timestamp(
    seconds: float,
    always_include_hours: bool = True,
    decimal_marker: str = ",",
) -> str:
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return (
        f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    )


def encode_audio_to_base64(audio_path: str) -> str:
    """
    Read audio file and return base64-encoded string.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    with open(audio_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def convert_audio(input_file, output_file, target_sr=16000, channels=2):
    """
    Convert audio to a given sample rate & channel count using ffmpeg.

    Args:
        input_file (str): Path to input audio file
        output_file (str): Path to save converted file
        target_sr (int): Target sample rate (e.g 16000)
        channels (int): Number of channels (1=mono, 2=stereo)
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_file,
        "-ar",
        str(target_sr),  # sample rate
        "-ac",
        str(channels),  # channels
        output_file,
    ]
    subprocess.run(cmd, check=True)


def restore_speech_timestamps(
    segments: List[Segment],
    speech_chunks: List[dict],
    sampling_rate: int = 16000,
    maintain_gaps: bool = False,
) -> List[Segment]:

    ts_map = SpeechTimestampsMap(
        speech_chunks, sampling_rate, maintain_gaps=maintain_gaps
    )

    for segment in segments:
        if segment.get("words"):
            words = []
            for word in segment["words"]:
                # Ensure the word start and end times are resolved to the same chunk.
                middle = (word["start"] + word["end"]) / 2
                chunk_index = ts_map.get_chunk_index(middle)
                word["start"] = ts_map.get_original_time(word["start"], chunk_index)
                word["end"] = ts_map.get_original_time(word["end"], chunk_index)
                words.append(word)

            segment["start"] = words[0]["start"]
            segment["end"] = words[-1]["end"]
            segment["words"] = words

        else:
            segment["start"] = ts_map.get_original_time(segment["start"])
            segment["end"] = ts_map.get_original_time(segment["end"])

    return segments


def convert_srt(output_file: str, segments: List) -> str:
    """Generate SRT content from segments, save to file, and return it."""
    if not output_file.endswith(".srt"):
        output_file += ".srt"

    # Build SRT content directly in memory
    srt_content = ""
    for i, s in enumerate(segments, start=1):
        srt_content += f"{i}\n"
        srt_content += (
            f"{format_timestamp(s['start'])} --> {format_timestamp(s['end'])}\n"
        )
        srt_content += f"{s['text'].strip()}\n\n"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(srt_content)

    return srt_content


def convert_txt(output_file: str, segments: List = [], txt_content: str = "") -> str:
    """Generate TXT content from segments, save to file, and return it."""
    if not output_file.endswith(".txt"):
        output_file += ".txt"
    if not txt_content:
        txt_content = ""
        for s in segments:
            txt_content += f"{s['text'].strip()}\n"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(txt_content)

    return txt_content


def assign_word_speakers(
    diarize_df,
    transcript_result: Dict[str, Any],
    fill_nearest: bool = False,
    start_pattern: str = "[{",
    end_pattern: str = "}]: ",
) -> Dict[str, List[Dict[str, Union[str, float]]]]:
    """
    Assigns speaker labels to transcript segments and words based on diarization,
    and formats the final transcript with speaker tags.
    """

    transcript_segments = transcript_result["segments"]

    # Handle Pydantic Segment objects (convert to dicts)
    if transcript_segments and not isinstance(transcript_segments[0], dict):
        transcript_segments = [seg.model_dump() for seg in transcript_segments]

    for seg in transcript_segments:
        # Assign speaker to segment (intersection logic)
        diarize_df["intersection"] = np.minimum(
            diarize_df["end"], seg["end"]
        ) - np.maximum(diarize_df["start"], seg["start"])
        diarize_df["union"] = np.maximum(diarize_df["end"], seg["end"]) - np.minimum(
            diarize_df["start"], seg["start"]
        )

        intersected = diarize_df[diarize_df["intersection"] > 0]
        speaker = None
        if len(intersected) > 0:
            speaker = (
                intersected.groupby("speaker")["intersection"]
                .sum()
                .sort_values(ascending=False)
                .index[0]
            )
        elif fill_nearest:
            speaker = diarize_df.sort_values(by=["intersection"], ascending=False)[
                "speaker"
            ].values[0]

        if speaker is not None:
            seg["speaker"] = speaker
        else:
            seg["speaker"] = "None"

        # Assign speaker to words inside the segment
        if "words" in seg and seg["words"] is not None:
            for word in seg["words"]:
                if "start" in word:
                    diarize_df["intersection"] = np.minimum(
                        diarize_df["end"], word["end"]
                    ) - np.maximum(diarize_df["start"], word["start"])

                    intersected = diarize_df[diarize_df["intersection"] > 0]
                    word_speaker = None
                    if len(intersected) > 0:
                        word_speaker = (
                            intersected.groupby("speaker")["intersection"]
                            .sum()
                            .sort_values(ascending=False)
                            .index[0]
                        )
                    elif fill_nearest:
                        word_speaker = diarize_df.sort_values(
                            by=["intersection"], ascending=False
                        )["speaker"].values[0]

                    if word_speaker is not None:
                        word["speaker"] = word_speaker
                    else:
                        word["speaker"] = "None"

    # Build final output with speaker-tagged text
    segments_result = []
    for seg in transcript_segments:
        speaker = seg.get("speaker", "None")
        diarized_text = f"{start_pattern}{speaker}{end_pattern}{seg['text'].strip()}"

        segments_result.append(
            {"start": seg["start"], "end": seg["end"], "text": diarized_text}
        )

    return segments_result


def derive_strip_chars(start_tag: str = "[{", end_tag: str = "}]: ") -> str:
    logger.debug(
        f"[derive_strip_chars] Called with start_tag='{start_tag}', end_tag='{end_tag}'"
    )
    chars = set(start_tag) | set(end_tag)
    result = "".join(chars)
    logger.debug(f"[derive_strip_chars] Derived strip chars: '{result}'")
    return result


def text_to_json(
    text: str,
    start_pattern: str = "[{",
    end_pattern: str = "}]: ",
    seperator: str = "|",
    robustness: bool = False,
) -> dict:
    logger.info(
        f"[text_to_json] Converting text to JSON | start_pattern='{start_pattern}', end_pattern='{end_pattern}', seperator='{seperator}', robustness={robustness}"
    )
    pattern = re.compile(rf"{re.escape(start_pattern)}(.*?){re.escape(end_pattern)}\s*")
    strip_chars = derive_strip_chars(start_pattern, end_pattern)
    matches = list(re.finditer(pattern, text))
    logger.debug(f"[text_to_json] Found {len(matches)} metadata matches in text")

    segments = []
    unique_speakers = set()
    last_end = 0
    current_meta = {"original_speaker": None, "speed": None, "pitch": None}

    for idx, match in enumerate(matches):
        logger.debug(
            f"[text_to_json] Processing match {idx+1}/{len(matches)} at position {match.start()}-{match.end()}"
        )
        text_segment = text[last_end : match.start()]
        if text_segment:
            logger.debug(
                f"[text_to_json] Adding text segment before match {idx+1} with length {len(text_segment)}"
            )
            segments.append({**current_meta, "original_text": text_segment})

        if robustness:
            parts = [p.strip(strip_chars) for p in match.group(1).split(seperator)]
        else:
            parts = match.group(1).split(seperator)

        original_speaker = parts[0] if len(parts) > 0 else None
        speed = parts[1] if len(parts) > 1 else None
        pitch = parts[2] if len(parts) > 2 else None
        logger.debug(
            f"[text_to_json] Extracted metadata | speaker='{original_speaker}', speed='{speed}', pitch='{pitch}'"
        )
        unique_speakers.add(original_speaker)
        current_meta = {
            "original_speaker": original_speaker,
            "speed": speed,
            "pitch": pitch,
        }
        last_end = match.end()

    trailing_text = text[last_end:]
    if trailing_text:
        logger.debug(
            f"[text_to_json] Adding trailing text segment with length {len(trailing_text)}"
        )
        segments.append({**current_meta, "original_text": trailing_text})

    segment_json = {"segments": segments, "unique_speakers": list(unique_speakers)}
    logger.info(
        f"[text_to_json] Completed parsing. Segments: {len(segments)}, Unique speakers: {len(unique_speakers)}"
    )
    return segment_json


def text_to_segments(
    text: str,
    start_pattern: str = "[{",
    end_pattern: str = "}]: ",
    seperator: str = "|",
    robustness: bool = False,
) -> list[dict]:
    logger.info(
        f"[text_to_segments] Splitting text into segments | start_pattern='{start_pattern}', end_pattern='{end_pattern}', seperator='{seperator}', robustness={robustness}"
    )
    pattern = re.compile(rf"{re.escape(start_pattern)}(.*?){re.escape(end_pattern)}\s*")
    strip_chars = derive_strip_chars(start_pattern, end_pattern)
    matches = list(re.finditer(pattern, text))
    logger.debug(f"[text_to_segments] Found {len(matches)} metadata matches in text")

    segments = []
    last_end = 0
    current_meta = {"original_speaker": None, "speed": None, "pitch": None}

    for idx, match in enumerate(matches):
        logger.debug(
            f"[text_to_segments] Processing match {idx+1}/{len(matches)} at position {match.start()}-{match.end()}"
        )
        text_segment = text[last_end : match.start()]
        if text_segment:
            logger.debug(
                f"[text_to_segments] Adding text segment before match {idx+1} with length {len(text_segment)}"
            )
            segments.append({**current_meta, "original_text": text_segment})

        if robustness:
            parts = [p.strip(strip_chars) for p in match.group(1).split(seperator)]
        else:
            parts = match.group(1).split(seperator)

        current_meta = {
            "original_speaker": parts[0] if len(parts) > 0 else None,
            "speed": parts[1] if len(parts) > 1 else None,
            "pitch": parts[2] if len(parts) > 2 else None,
        }
        logger.debug(f"[text_to_segments] Extracted metadata | {current_meta}")
        last_end = match.end()

    trailing_text = text[last_end:]
    if trailing_text:
        logger.debug(
            f"[text_to_segments] Adding trailing text segment with length {len(trailing_text)}"
        )
        segments.append({**current_meta, "original_text": trailing_text})

    logger.info(f"[text_to_segments] Completed parsing. Segments: {len(segments)}")
    return segments


def srt_to_json(
    text: str,
    start_pattern: str = "[{",
    end_pattern: str = "}]: ",
    seperator: str = "|",
    robustness: bool = False,
) -> dict:
    logger.info(
        f"[srt_to_json] Converting SRT text to JSON | start_pattern='{start_pattern}', end_pattern='{end_pattern}', seperator='{seperator}', robustness={robustness}"
    )
    segments = []
    unique_speakers = set()
    subs = pysrt.from_string(text)
    logger.debug(f"[srt_to_json] Parsed {len(subs)} subtitle entries from SRT text")

    for idx, sub in enumerate(subs):
        logger.debug(
            f"[srt_to_json] Processing subtitle {idx+1}/{len(subs)} with time {sub.start} --> {sub.end}"
        )
        local_segments = text_to_segments(
            sub.text, start_pattern, end_pattern, seperator, robustness
        )
        logger.debug(
            f"[srt_to_json] Extracted {len(local_segments)} segments from subtitle {idx+1}"
        )
        for seg in local_segments:
            seg["start_time"] = str(sub.start)
            seg["end_time"] = str(sub.end)
            segments.append(seg)
            unique_speakers.add(seg["original_speaker"])

    segment_json = {"segments": segments, "unique_speakers": list(unique_speakers)}
    logger.info(
        f"[srt_to_json] Completed parsing. Segments: {len(segments)}, Unique speakers: {len(unique_speakers)}"
    )
    return segment_json


def stt_pipeline(
    stt_service: str,
    start_pattern: str,
    end_pattern: str,
    language: str,
    wlt: bool,
    uvr: bool,
    vad: bool,
    diarize: bool,
    req_id: str,
    output_format: str = "srt",
    audio_url: str = "",
    audio_path: str = "",
    maintain_gaps: bool = False,
):
    """Main pipeline to process audio → VAD → STT → diarization (optional) → SRT."""

    if not (audio_path and os.path.exists(audio_path)):
        # ---------------------------
        # Step 1: Download audio & encode base64
        # ---------------------------
        logger.info(f"[{req_id}] Downloading audio from {audio_url}")
        audio_response = requests.get(audio_url)
        if audio_response.status_code != 200:
            return {"status": "error", "message": "Failed to download audio file"}

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_response.content)
            audio_path = Path(tmp.name)

    # audio_16khz = "audio_16khz_" + audio_path
    # convert_audio(audio_path, audio_16khz, 16000, 1)
    # audio_base64 = encode_audio_to_base64(str(audio_16khz))
    audio_base64 = encode_audio_to_base64(str(audio_path))

    # ---------------------------
    # Step 3: Run UVR
    # ---------------------------
    if uvr:
        uvr_payload = {
            "audio_base64": audio_base64,
            "req_id": req_id,
        }
        uvr_response = requests.post(UVR_URL, json=uvr_payload).json()
        if uvr_response.get("status") != "success":
            return {"status": "error", "message": "UVR failed"}

        audio_base64 = uvr_response["vocal_base64"]

    # ---------------------------
    # Step 3: Run VAD
    # ---------------------------
    vad_audio_base64 = None
    if vad:
        vad_payload = {
            "audio_base64": audio_base64,
            "sample_rate": 16000,
            "req_id": req_id,
        }
        vad_response = requests.post(VAD_URL, json=vad_payload).json()
        if vad_response.get("status") != "success":
            return {"status": "error", "message": "VAD failed"}

        speech_timestamps = vad_response["speech_timestamps"]
        vad_audio_base64 = vad_response["audio_base64"]
    # ---------------------------
    # Step 4: Run ASR
    # ---------------------------
    asr_payload = {
        "audio_base64": vad_audio_base64 if vad else audio_base64,
        "language": language,
        "wlt": wlt,
        "req_id": req_id,
    }

    if stt_service == "whisper":
        asr_response = requests.post(WHISPER_URL, json=asr_payload).json()
    elif stt_service == "faster-whisper":
        asr_response = requests.post(FASTER_WHISPER_URL, json=asr_payload).json()
    elif stt_service == "indic-conformer":
        if language not in INDIC_CONFORMER_LANGUAGES:
            return {
                "status": "error",
                "message": "Please provide Supported Languages for Indic Conformer Service",
            }
        asr_response = requests.post(INDIC_CONFORMER_URL, json=asr_payload).json()
    else:
        return {"status": "error", "message": "Invalid STT service provided"}

    if asr_response.get("status") != "success":
        return {"status": "error", "message": "ASR failed"}

    segments = asr_response["data"]

    # ---------------------------
    # Step 5: Restore timestamps
    # ---------------------------
    restored_result = None
    if vad and stt_service != "indic-conformer":
        restored_result = restore_speech_timestamps(
            segments=segments,
            speech_chunks=speech_timestamps,
            sampling_rate=16000,
            maintain_gaps=maintain_gaps,
        )

    # ---------------------------
    # Step 6: Diarization
    # ---------------------------
    if diarize and stt_service != "indic-conformer":
        diarization_payload = {"audio_base64": audio_base64, "req_id": req_id}
        diarization_response = requests.post(
            DIARIZE_URL, json=diarization_payload
        ).json()
        if diarization_response.get("status") != "success":
            return {"status": "error", "message": "Diarization failed"}

        diarize_df = pd.DataFrame(diarization_response["data"])
        segments_result = assign_word_speakers(
            diarize_df=diarize_df,
            transcript_result={"segments": restored_result if vad else segments},
            start_pattern=start_pattern,
            end_pattern=end_pattern,
        )
    else:
        segments_result = restored_result if restored_result else segments

    # ---------------------------
    # Step 7: Convert to SRT
    # ---------------------------
    output_path = f"{tempfile.gettempdir()}/{uuid.uuid4()}_{req_id}_output"
    if stt_service == "indic-conformer":
        output_path += ".txt"
        content = convert_txt(output_path, txt_content=segments_result)
    elif output_format == "srt":
        output_path += ".srt"
        content = convert_srt(output_path, segments_result)
    elif output_format == "txt":
        output_path += ".txt"
        content = convert_txt(output_path, segments_result)
    else:
        content = ""

    logger.info(f"[{req_id}] Pipeline completed successfully.")

    if diarize and stt_service != "indic-conformer":
        logger.info(
            f"[transcribe_and_parse] Parsing transcript with diarization enabled | format='{output_format}'"
        )
        if str(output_format).lower() == "txt":
            parsed_json = text_to_json(
                text=content,
                start_pattern=start_pattern,
                end_pattern=end_pattern,
            )
        elif str(output_format).lower() == "srt":
            parsed_json = srt_to_json(
                text=content,
                start_pattern=start_pattern,
                end_pattern=end_pattern,
            )
        logger.info(
            f"[transcribe_and_parse] Parsing successful | segments={len(parsed_json['segments'])}, unique_speakers={len(parsed_json['unique_speakers'])}"
        )

        return {
            "status": "success",
            "req_id": req_id,
            "transcript": content,
            "output_file": str(output_path),
            "segments": parsed_json["segments"],
            "original_speakers": parsed_json["unique_speakers"],
        }
    else:
        logger.info(
            "[transcribe_and_parse] Returning transcription without diarization"
        )
        return {
            "status": "success",
            "req_id": req_id,
            "output_file": output_path,
            "transcript": content,
        }
