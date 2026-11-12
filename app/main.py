import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.routers import chat, health, stt, tts, ws
from app.services.brain import generate_response
from app.services.conversation import (
    close_db,
    get_history,
    init_db,
    save_exchange,
)
from app.services.stt import load_stt_model, transcribe_audio
from app.services.tts import load_tts_model, tts_lock
from app.services.brain import check_brain, generate_response


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading Hivi models...")

    app.state.tts_model, app.state.voice_state = load_tts_model()
    app.state.stt_model = load_stt_model()

    await init_db()

    try:
        await check_brain(timeout=120)
        print("[BRAIN] Model Ollama siap.")
    except Exception as error:
        print(f"[BRAIN] Warmup gagal (server tetap jalan): {error!r}")

    try:
        # Konfigurasi resource WebSocket
        app.state.ws_connection_slots = asyncio.Semaphore(3)
        app.state.audio_processing_slots = asyncio.Semaphore(1)
        app.state.active_devices = set()

        # Service yang dipakai handler WebSocket
        app.state.stt_transcribe = transcribe_audio
        app.state.generate_response = generate_response
        app.state.get_history = get_history
        app.state.save_exchange = save_exchange
        app.state.tts_lock = tts_lock

        print("Hivi is ready!")
        yield
    finally:
        await close_db()
        print("Shutting down Hivi...")


app = FastAPI(
    title="Hivi API",
    description="Indonesian voice assistant service for Hivi",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(tts.router)
app.include_router(stt.router)
app.include_router(chat.router)
app.include_router(ws.router)