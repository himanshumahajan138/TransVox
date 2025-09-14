import os
import io
import base64
import tempfile
import subprocess
import soundfile as sf
import numpy as np
from typing import Union
from scipy.io.wavfile import write


def load_audio(file: Union[str, np.ndarray], sr: int = 16000) -> np.ndarray:
    if isinstance(file, np.ndarray):
        if file.dtype != np.float32:
            file = file.astype(np.float32)
        if file.ndim > 1:
            file = np.mean(file, axis=1)

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        write(temp_file.name, sr, (file * 32768).astype(np.int16))
        temp_file_path = temp_file.name
        temp_file.close()
    else:
        temp_file_path = file

    try:
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-threads",
            "0",
            "-i",
            temp_file_path,
            "-f",
            "s16le",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sr),
            "-",
        ]
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    finally:
        if isinstance(file, np.ndarray):
            os.remove(temp_file_path)

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
