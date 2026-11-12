from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.services.stt import transcribe_audio

router = APIRouter(prefix="/api/v1", tags=["stt"])

ALLOWED_CONTENT_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/vnd.wave",
    "application/octet-stream",
}
MAX_AUDIO_BYTES = 10 * 1024 * 1024  # 10 MB


@router.post("/stt")
def generate_stt(request: Request, audio: UploadFile = File(...)):
    if audio.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Only WAV audio files are supported")

    try:
        audio.file.seek(0, 2)
        file_size = audio.file.tell()
        audio.file.seek(0)

        if file_size == 0:
            raise HTTPException(status_code=400, detail="Audio file is empty")

        if file_size > MAX_AUDIO_BYTES:
            raise HTTPException(status_code=413, detail="Audio file exceeds 10 MB")

        result = transcribe_audio(request.app.state.stt_model, audio.file)

        if not result["text"]:
            raise HTTPException(status_code=422, detail="No speech detected")

        return {"service": "hivi-stt", "status": "success", "data": result}

    except HTTPException:
        raise

    except Exception as error:
        print(f"STT generation error: {error}")
        raise HTTPException(status_code=500, detail="Failed to transcribe audio")

    finally:
        audio.file.close()