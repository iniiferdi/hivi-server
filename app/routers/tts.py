from io import BytesIO

import numpy as np
import scipy.io.wavfile
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from app.schemas import TTSRequest
from app.services.tts import generate_pcm16

router = APIRouter(prefix="/api/v1", tags=["tts"])


@router.post("/tts")
def generate_tts(payload: TTSRequest, request: Request):
    state = request.app.state

    try:
        pcm16 = generate_pcm16(
            state.tts_model,
            state.voice_state,
            payload.text,
            state.tts_lock,
        )

        buffer = BytesIO()
        scipy.io.wavfile.write(
            buffer,
            state.tts_model.sample_rate,
            np.frombuffer(pcm16, dtype="<i2"),
        )

        return Response(
            content=buffer.getvalue(),
            media_type="audio/wav",
            headers={"Content-Disposition": 'attachment; filename="hivi.wav"'},
        )

    except Exception as error:
        print(f"TTS generation error: {error}")
        raise HTTPException(status_code=500, detail="Failed to generate audio")