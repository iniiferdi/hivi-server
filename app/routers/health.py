from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
def health(request: Request):
    state = request.app.state

    tts_ready = hasattr(state, "tts_model") and hasattr(state, "voice_state")
    stt_ready = hasattr(state, "stt_model")

    return {
        "service": "hivi-server",
        "status": "ready" if tts_ready and stt_ready else "starting",
        "services": {
            "tts": "ready" if tts_ready else "unavailable",
            "stt": "ready" if stt_ready else "unavailable",
        },
    }