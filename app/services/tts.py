
import os
from collections import OrderedDict
from collections.abc import Iterator
from threading import Lock

import numpy as np
import torch
from pocket_tts import TTSModel

from app.config import (
    TTS_CACHE_MAX_ENTRIES,
    TTS_MODEL_CONFIG,
    TTS_NUM_THREADS,
    TTS_WARMUP_TEXT,
    VOICE_PATH,
)

# Cegah generate bersamaan pada satu instance model
tts_lock = Lock()

# Cache PCM16 untuk kalimat yang sering diulang (LRU)
_pcm_cache: OrderedDict[str, bytes] = OrderedDict()
_cache_lock = Lock()


def _to_pcm16(audio) -> bytes:
    """Konversi tensor/array audio menjadi bytes PCM16 little-endian mono."""
    audio_np = (
        audio.detach().cpu().numpy()
        if hasattr(audio, "detach")
        else audio
    )
    audio_np = np.asarray(audio_np, dtype=np.float32).squeeze()

    if audio_np.size == 0:
        return b""

    audio_np = np.nan_to_num(
        audio_np,
        nan=0.0,
        posinf=1.0,
        neginf=-1.0,
    )
    audio_np = np.clip(audio_np, -1.0, 1.0)

    return np.rint(audio_np * 32767.0).astype("<i2").tobytes()


def load_tts_model():
    if not VOICE_PATH.exists():
        raise RuntimeError(f"Voice prompt not found: {VOICE_PATH}")

    torch.set_num_threads(TTS_NUM_THREADS or os.cpu_count() or 1)

    model = TTSModel.load_model(config=TTS_MODEL_CONFIG)
    voice_state = model.get_state_for_audio_prompt(str(VOICE_PATH))

    # Warmup agar request pertama tidak menanggung biaya inisialisasi
    with torch.no_grad():
        model.generate_audio(voice_state, TTS_WARMUP_TEXT)

    return model, voice_state


def stream_pcm16(
    model,
    voice_state,
    text,
    lock=tts_lock,
) -> Iterator[bytes]:
    """Hasilkan audio mono PCM16 per chunk begitu tersedia."""

    with lock, torch.no_grad():
        produced = False

        for chunk in model.generate_audio_stream(voice_state, text):
            pcm = _to_pcm16(chunk)

            if pcm:
                produced = True
                yield pcm

    if not produced:
        raise RuntimeError("TTS menghasilkan audio kosong.")


def generate_pcm16(
    model,
    voice_state,
    text,
    lock=tts_lock,
) -> bytes:
    """Generate mono PCM16 audio utuh, dengan cache untuk teks berulang."""

    with _cache_lock:
        cached = _pcm_cache.get(text)

        if cached is not None:
            _pcm_cache.move_to_end(text)
            return cached

    with lock, torch.no_grad():
        audio = model.generate_audio(voice_state, text)

    pcm = _to_pcm16(audio)

    if not pcm:
        raise RuntimeError("TTS menghasilkan audio kosong.")

    with _cache_lock:
        _pcm_cache[text] = pcm
        _pcm_cache.move_to_end(text)

        while len(_pcm_cache) > TTS_CACHE_MAX_ENTRIES:
            _pcm_cache.popitem(last=False)

    return pcm
