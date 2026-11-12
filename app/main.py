
from contextlib import asynccontextmanager
from pathlib import Path
from io import BytesIO
from threading import Lock


import asyncio
import numpy as np
import scipy.io.wavfile
import torch

from app.services.brain import generate_response

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from app.services.stt import load_stt_model, transcribe_audio

from fastapi.responses import Response
from pydantic import BaseModel, Field
from pocket_tts import TTSModel

from app.services.conversation import (
    init_redis,
    close_redis,
    get_history,
    save_exchange,
    clear_session,
)

from fastapi import WebSocket

from app.services.audio_stream import handle_audio_stream

from uuid import uuid4


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
VOICE_PATH = BASE_DIR / "voices" / "voice_1.wav"

MODEL_CONFIG = (
    "hf://anak10thn/pocket-tts-indonesian/"
    "indonesian_6l.yaml@17257664e384561c957b02ac92edd1a24807f0e5"
)

# Prevent simultaneous generation on the same model instance
tts_lock = Lock()


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None

class DeviceCommand(BaseModel):
    command: str = Field(min_length=1, max_length=50)
    parameters: dict = Field(default_factory=dict)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading Indonesian Pocket TTS...")

    if not VOICE_PATH.exists():
        raise RuntimeError(f"Voice prompt not found: {VOICE_PATH}")

    model = TTSModel.load_model(config=MODEL_CONFIG)
    voice_state = model.get_state_for_audio_prompt(str(VOICE_PATH))

    stt_model = load_stt_model()
    app.state.stt_model = stt_model

    app.state.tts_model = model
    app.state.voice_state = voice_state

    

    await init_redis()

    # WebSocket resource configuration
    app.state.ws_connection_slots = asyncio.Semaphore(3)
    app.state.audio_processing_slots = asyncio.Semaphore(1)
    app.state.active_devices = set()

    # Share existing services with WebSocket handler
    app.state.stt_transcribe = transcribe_audio
    app.state.generate_response = generate_response
    app.state.get_history = get_history
    app.state.save_exchange = save_exchange
    app.state.tts_lock = tts_lock

    print("Hivi TTS is ready!")

    yield
    await close_redis()

    print("Shutting down Hivi TTS...")


app = FastAPI(
    title="Hivi TTS API",
    description="Indonesian Text-to-Speech service for Hivi",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request):
    tts_ready = hasattr(request.app.state, "tts_model")
    stt_ready = hasattr(request.app.state, "stt_model")
    voice_ready = hasattr(request.app.state, "voice_state")

    ready = tts_ready and stt_ready and voice_ready

    return {
        "service": "hivi-server",
        "status": "ready" if ready else "starting",
        "services": {
            "tts": "ready" if tts_ready and voice_ready else "unavailable",
            "stt": "ready" if stt_ready else "unavailable"
        }
    }


@app.post("/api/v1/tts")
def generate_tts(payload: TTSRequest, request: Request):
    model = request.app.state.tts_model
    voice_state = request.app.state.voice_state

    try:
        with tts_lock:
            audio = model.generate_audio(
                voice_state,
                payload.text
            )

        audio_np = audio.detach().cpu().numpy()
        audio_np = np.asarray(audio_np, dtype=np.float32).squeeze()

        if audio_np.size == 0:
            raise ValueError("Generated audio is empty")

        audio_np = np.clip(audio_np, -1.0, 1.0)

        buffer = BytesIO()

        scipy.io.wavfile.write(
            buffer,
            model.sample_rate,
            audio_np
        )

        return Response(
            content=buffer.getvalue(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": 'attachment; filename="hivi.wav"'
            },
        )

    except Exception as error:
        print(f"TTS generation error: {error}")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate audio"
        )



@app.post("/api/v1/stt")
def generate_stt(
    request: Request,
    audio: UploadFile = File(...)
):
    if audio.content_type not in {
        "audio/wav",
        "audio/x-wav",
        "audio/wave",
        "audio/vnd.wave",
        "application/octet-stream"
    }:
        raise HTTPException(
            status_code=415,
            detail="Only WAV audio files are supported"
        )

    try:
        # Batasi ukuran audio maksimal 10 MB
        audio.file.seek(0, 2)
        file_size = audio.file.tell()
        audio.file.seek(0)

        if file_size == 0:
            raise HTTPException(
                status_code=400,
                detail="Audio file is empty"
            )

        if file_size > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Audio file exceeds 10 MB"
            )

        stt_model = request.app.state.stt_model

        result = transcribe_audio(
            stt_model,
            audio.file
        )

        if not result["text"]:
            raise HTTPException(
                status_code=422,
                detail="No speech detected"
            )

        return {
            "service": "hivi-stt",
            "status": "success",
            "data": result
        }

    except HTTPException:
        raise

    except Exception as error:
        print(f"STT generation error: {error}")

        raise HTTPException(
            status_code=500,
            detail="Failed to transcribe audio"
        )

    finally:
        audio.file.close()



@app.post("/api/v1/chat")
async def chat(payload: ChatRequest):
    session_id = payload.session_id or str(uuid4())

    try:
        # Ambil riwayat percakapan dari Redis
        history = await get_history(session_id)

        # Kirim pesan ke Gemini
        answer = await generate_response(
            message=payload.message,
            history=history,
        )

        # Simpan percakapan ke Redis
        await save_exchange(
            session_id=session_id,
            user_message=payload.message,
            assistant_message=answer,
        )

        return {
            "service": "hivi-brain",
            "session_id": session_id,
            "message": payload.message,
            "response": answer,
        }

    except Exception as error:
        print(f"AI Brain error: {error}")
        raise HTTPException(
            status_code=502,
            detail="AI Brain sedang mengalami gangguan.",
        )

@app.websocket("/ws/audio/{device_id}")
async def websocket_audio(
    websocket: WebSocket,
    device_id: str,
):
    await handle_audio_stream(websocket, device_id)
