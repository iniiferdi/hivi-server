import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
VOICE_PATH = BASE_DIR / "voices" / "hivi_vintage.wav"

TTS_MODEL_CONFIG = (
    "hf://anak10thn/pocket-tts-indonesian/"
    "indonesian_6l.yaml@17257664e384561c957b02ac92edd1a24807f0e5"
)
TTS_WARMUP_TEXT = "Halo, aku Hivi."
TTS_NUM_THREADS = int(os.getenv("HIVI_TTS_THREADS", "0"))  # 0 = otomatis
TTS_CACHE_MAX_ENTRIES = 64

# STT (Faster-Whisper)
STT_MODEL_SIZE = os.getenv("HIVI_STT_MODEL", "tiny")
STT_CPU_THREADS = int(os.getenv("HIVI_STT_THREADS", "4"))
STT_LANGUAGE = "id"
STT_BEAM_SIZE = 1

HIVI_WS_TOKEN = os.getenv("HIVI_WS_TOKEN", "")

# WebSocket
WS_MAX_CONNECTIONS = 3
WS_AUTH_TIMEOUT = 5
WS_IDLE_TIMEOUT = 15
WS_MAX_CHUNK_BYTES = 8192

# Audio input
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_MAX_SIZE = 10 * 1024 * 1024
AUDIO_MAX_DURATION = 120

# Pemrosesan audio
AUDIO_MAX_CONCURRENT_PROCESSING = 1