from threading import Lock

import numpy as np
from pocket_tts import TTSModel

from app.config import TTS_MODEL_CONFIG, VOICE_PATH

# Cegah generate bersamaan pada satu instance model
tts_lock = Lock()


def load_tts_model():
    if not VOICE_PATH.exists():
        raise RuntimeError(f"Voice prompt not found: {VOICE_PATH}")

    model = TTSModel.load_model(config=TTS_MODEL_CONFIG)
    voice_state = model.get_state_for_audio_prompt(str(VOICE_PATH))

    return model, voice_state


def generate_pcm16(model, voice_state, text, lock=tts_lock) -> bytes:
    """Generate mono PCM16 audio using Pocket TTS."""

    with lock:
        audio = model.generate_audio(voice_state, text)

    audio_np = audio.detach().cpu().numpy()
    audio_np = np.asarray(audio_np, dtype=np.float32).squeeze()

    if audio_np.size == 0:
        raise RuntimeError("TTS menghasilkan audio kosong.")

    audio_np = np.nan_to_num(audio_np, nan=0.0, posinf=1.0, neginf=-1.0)
    audio_np = np.clip(audio_np, -1.0, 1.0)

    return np.rint(audio_np * 32767.0).astype("<i2").tobytes()