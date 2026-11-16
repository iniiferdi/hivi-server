import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import WS_MAX_CONNECTIONS
from app.routers import health, stt, tts, ws
from app.services.conversation import (
    close_db,
    get_history,
    init_db,
    save_exchange,
)
from app.services.stt import load_stt_model, transcribe_samples
from app.services.tts import load_tts_model, tts_lock

logging.basicConfig(
    level=os.getenv("HIVI_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

# Brain (Ollama) dimatikan dulu. Nyalakan lagi dengan: HIVI_BRAIN_ENABLED=1
BRAIN_ENABLED = os.getenv("HIVI_BRAIN_ENABLED", "0") == "1"

BRAIN_OFF_REPLY = "Fitur otak Hivi sedang dimatikan."


async def _brain_disabled_response(*args, **kwargs):
    # Pengganti generate_response saat brain mati: tidak memanggil Ollama sama sekali.
    return BRAIN_OFF_REPLY


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading Hivi models...")

    # Model TTS dan STT dimuat paralel agar startup lebih singkat.
    (app.state.tts_model, app.state.voice_state), app.state.stt_model = (
        await asyncio.gather(
            asyncio.to_thread(load_tts_model),
            asyncio.to_thread(load_stt_model),
        )
    )

    await init_db()

    if BRAIN_ENABLED:
        # Import di sini agar paket 'ollama' hanya dibutuhkan bila brain menyala
        from app.services.brain import check_brain, generate_response

        try:
            await check_brain(timeout=120)
            logger.info("[BRAIN] Model Ollama siap.")
        except Exception as error:
            logger.warning("[BRAIN] Warmup gagal (server tetap jalan): %r", error)
        brain_generate = generate_response
    else:
        logger.info("[BRAIN] Dimatikan (set HIVI_BRAIN_ENABLED=1 untuk menyalakan).")
        brain_generate = _brain_disabled_response

    try:
        # Batas koneksi WebSocket
        app.state.ws_connection_slots = asyncio.Semaphore(WS_MAX_CONNECTIONS)

        # Service yang dipakai handler WebSocket
        app.state.stt_transcribe = transcribe_samples
        app.state.generate_response = brain_generate
        app.state.get_history = get_history
        app.state.save_exchange = save_exchange
        app.state.tts_lock = tts_lock

        logger.info("Hivi is ready!")
        yield
    finally:
        await close_db()
        logger.info("Shutting down Hivi...")


app = FastAPI(
    title="Hivi API",
    description="Indonesian voice assistant service for Hivi",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(tts.router)
app.include_router(stt.router)
if BRAIN_ENABLED:
    # chat.py meng-import brain.py (dan paket ollama), jadi hanya dimuat bila brain menyala
    from app.routers import chat
    app.include_router(chat.router)
app.include_router(ws.router)