"""Decode MP3 and integer PCM WAV without extra Python dependencies.

PCM stays in its original sample rate and channel layout. Returning frames x
channels lets timing analysis combine channel evidence without cancelling
opposite-phase stereo. MP3 uses the vendored MIT-0 dr_mp3 through stdlib ctypes;
the tiny bridge builds locally once when needed.
"""
from __future__ import annotations

import ctypes
from pathlib import Path
import threading
import wave

import numpy as np

from tools.build_audio_decoder import DecoderBuildError, build_decoder


class AudioDecodeError(RuntimeError):
    """An audio file is unsupported, damaged, empty, or cannot be decoded."""


_decoder = None
_decoder_lock = threading.Lock()


def _native_decoder():
    global _decoder
    with _decoder_lock:
        if _decoder is None:
            try:
                decoder = ctypes.CDLL(str(build_decoder()))
            except (DecoderBuildError, OSError) as exc:
                raise AudioDecodeError(str(exc)) from exc
            decoder.gm_decode_mp3.argtypes = [
                ctypes.c_void_p, ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint),
                ctypes.POINTER(ctypes.c_uint64),
            ]
            decoder.gm_decode_mp3.restype = ctypes.POINTER(ctypes.c_float)
            decoder.gm_free_pcm.argtypes = [ctypes.POINTER(ctypes.c_float)]
            decoder.gm_free_pcm.restype = None
            _decoder = decoder
        return _decoder


def _decode_mp3(path: Path) -> tuple[np.ndarray, int]:
    # Python owns file I/O, so non-ASCII filenames also work on Windows.
    content = path.read_bytes()
    if not content:
        raise AudioDecodeError("The MP3 file is empty.")
    decoder = _native_decoder()
    encoded = ctypes.create_string_buffer(content)
    channels, rate, frames = ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint64()
    pcm = decoder.gm_decode_mp3(encoded, len(content), ctypes.byref(channels),
                                ctypes.byref(rate), ctypes.byref(frames))
    if not pcm:
        raise AudioDecodeError("No valid MPEG audio frames could be decoded from this MP3.")
    try:
        if channels.value not in (1, 2) or not frames.value or not rate.value:
            raise AudioDecodeError("MP3 decoder returned invalid PCM metadata.")
        samples = np.ctypeslib.as_array(pcm, shape=(frames.value * channels.value,))
        # Copy before freeing memory allocated by the native decoder.
        audio = samples.reshape(-1, channels.value).copy()
    finally:
        decoder.gm_free_pcm(pcm)
    return audio, rate.value


def _decode_wav(path: Path) -> tuple[np.ndarray, int]:
    try:
        with wave.open(str(path), "rb") as source:
            channels = source.getnchannels()
            width = source.getsampwidth()
            rate = source.getframerate()
            frames = source.getnframes()
            if source.getcomptype() != "NONE" or width not in (1, 2, 3, 4):
                raise AudioDecodeError("WAV must contain uncompressed 8/16/24/32-bit integer PCM.")
            if not channels or not rate or not frames:
                raise AudioDecodeError("The WAV file contains no PCM frames.")
            content = source.readframes(frames)
    except (wave.Error, EOFError) as exc:
        raise AudioDecodeError(
            "Cannot decode this WAV; use uncompressed 8/16/24/32-bit integer PCM."
        ) from exc
    if len(content) != frames * channels * width:
        raise AudioDecodeError("The WAV file is truncated or has invalid frame data.")
    if width == 1:
        audio = (np.frombuffer(content, dtype=np.uint8).astype(np.float32) - 128) / 128
    elif width == 3:
        octets = np.frombuffer(content, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        values = octets[:, 0] | (octets[:, 1] << 8) | (octets[:, 2] << 16)
        values = (values ^ 0x800000) - 0x800000
        audio = values.astype(np.float32) / 8388608
    else:
        dtype = "<i2" if width == 2 else "<i4"
        audio = np.frombuffer(content, dtype=dtype).astype(np.float32) / float(2 ** (8 * width - 1))
    return audio.reshape(-1, channels), rate


def decode_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Return float32 PCM ``(frames, channels)`` and the original sample rate.

    Raises AudioDecodeError on unsupported/corrupt audio or missing MP3 compiler;
    normal filesystem errors (missing file, permission denied) remain OSError.
    MP3 may exceed +/-1 slightly due to reconstruction overshoot; do not clip it.
    """
    path = Path(path)
    extension = path.suffix.lower()
    if extension == ".wav":
        audio, rate = _decode_wav(path)
    elif extension == ".mp3":
        audio, rate = _decode_mp3(path)
    else:
        raise AudioDecodeError(f"Unsupported audio format {extension or '(no extension)'}; use MP3 or PCM WAV.")
    if audio.ndim != 2 or audio.shape[0] == 0 or not np.isfinite(audio).all():
        raise AudioDecodeError("Decoded audio is empty or contains invalid PCM samples.")
    return audio, int(rate)
