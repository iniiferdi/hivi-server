import logging
import wave
from io import BytesIO

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from app.schemas import TTSRequest
from app.services.tts import generate_pcm16, stream_pcm16

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["tts"])

AUDIO_CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2  # PCM16


def _pcm16_to_wav(pcm16: bytes, sample_rate: int) -> bytes:
    """Bungkus PCM16 mono menjadi file WAV."""
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(AUDIO_CHANNELS)
        wav.setsampwidth(SAMPLE_WIDTH_BYTES)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm16)
    return buffer.getvalue()


@router.post("/tts")
def generate_tts(payload: TTSRequest, request: Request):
    """Hasilkan audio WAV utuh untuk satu teks."""
    state = request.app.state

    try:
        pcm16 = generate_pcm16(
            state.tts_model,
            state.voice_state,
            payload.text,
            state.tts_lock,
        )
        wav_bytes = _pcm16_to_wav(pcm16, state.tts_model.sample_rate)
    except Exception:
        logger.exception("TTS generation error")
        raise HTTPException(status_code=500, detail="Failed to generate audio")

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="hivi.wav"'},
    )


@router.post("/tts/stream")
def stream_tts(payload: TTSRequest, request: Request):
    """Kirim audio PCM16 mono (little-endian) per chunk begitu tersedia."""
    state = request.app.state

    return StreamingResponse(
        stream_pcm16(
            state.tts_model,
            state.voice_state,
            payload.text,
            state.tts_lock,
        ),
        media_type="application/octet-stream",
        headers={
            "X-Audio-Format": "pcm_s16le",
            "X-Sample-Rate": str(state.tts_model.sample_rate),
            "X-Audio-Channels": str(AUDIO_CHANNELS),
        },
    )