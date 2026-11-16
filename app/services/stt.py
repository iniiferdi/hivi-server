import logging
from threading import Lock

import numpy as np
from faster_whisper import WhisperModel, decode_audio

from app.config import (
    AUDIO_MAX_DURATION,
    AUDIO_SAMPLE_RATE,
    STT_BEAM_SIZE,
    STT_CPU_THREADS,
    STT_LANGUAGE,
    STT_MODEL_SIZE,
)

logger = logging.getLogger(__name__)

# Cegah transkripsi bersamaan pada satu instance model
stt_lock = Lock()

_TRANSCRIBE_OPTIONS = {
    "language": STT_LANGUAGE,
    "beam_size": STT_BEAM_SIZE,
    "temperature": 0.0,  # satu kali decoding, tanpa fallback berulang
    "without_timestamps": True,
    "condition_on_previous_text": False,
    "vad_filter": True,
    "vad_parameters": {"min_silence_duration_ms": 300, "speech_pad_ms": 200},
}


class InvalidAudioError(ValueError):
    """Audio tidak bisa didekode."""


class AudioTooLongError(ValueError):
    """Durasi audio melebihi batas."""


def load_stt_model() -> WhisperModel:
    logger.info("Loading Faster-Whisper STT (%s)...", STT_MODEL_SIZE)

    model = WhisperModel(
        STT_MODEL_SIZE,
        device="cpu",
        compute_type="int8",
        cpu_threads=STT_CPU_THREADS,
        num_workers=1,
    )

    _warmup(model)
    logger.info("Hivi STT is ready!")

    return model


def _warmup(model: WhisperModel) -> None:
    """Jalankan satu transkripsi singkat agar request pertama tidak lambat."""
    silence = np.zeros(AUDIO_SAMPLE_RATE, dtype=np.float32)
    segments, _ = model.transcribe(
        silence,
        language=STT_LANGUAGE,
        beam_size=1,
        without_timestamps=True,
        vad_filter=False,
    )
    list(segments)


def _transcribe(model: WhisperModel, audio: np.ndarray) -> dict:
    """Transkripsi array float32 mono 16 kHz."""
    duration = len(audio) / AUDIO_SAMPLE_RATE

    if duration > AUDIO_MAX_DURATION:
        raise AudioTooLongError(
            f"Durasi audio {duration:.1f}s melebihi batas {AUDIO_MAX_DURATION}s."
        )

    with stt_lock:
        segments, info = model.transcribe(audio, **_TRANSCRIBE_OPTIONS)

        # Transkripsi bersifat lazy, jadi harus dikonsumsi
        # sebelum lock dilepaskan.
        transcription = " ".join(
            text for text in (segment.text.strip() for segment in segments) if text
        )

    return {
        "text": transcription,
        "language": info.language,
        "duration": round(duration, 2),
    }


def transcribe_audio(model: WhisperModel, audio_file) -> dict:
    """Transkripsi dari file audio (WAV) yang di-upload."""
    audio_file.seek(0)

    # Dekode di luar lock supaya request yang antre sudah siap saat gilirannya tiba.
    try:
        audio = decode_audio(audio_file, sampling_rate=AUDIO_SAMPLE_RATE)
    except Exception as error:
        raise InvalidAudioError("Audio tidak dapat didekode.") from error

    return _transcribe(model, audio)


def transcribe_samples(model: WhisperModel, samples: np.ndarray) -> dict:
    """Transkripsi dari sampel PCM16 mono 16 kHz, tanpa encode/decode WAV."""
    audio = samples.astype(np.float32) / 32768.0

    return _transcribe(model, audio)