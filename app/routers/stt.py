import logging

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.config import AUDIO_MAX_SIZE
from app.services.stt import AudioTooLongError, InvalidAudioError, transcribe_audio

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["stt"])

ALLOWED_CONTENT_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/vnd.wave",
    "application/octet-stream",
}
WAV_HEADER_SIZE = 12


def _is_wav(file) -> bool:
    """Cek header RIFF/WAVE, tidak hanya mengandalkan content-type."""
    header = file.read(WAV_HEADER_SIZE)
    file.seek(0)
    return header[:4] == b"RIFF" and header[8:12] == b"WAVE"


def _file_size(file) -> int:
    file.seek(0, 2)
    size = file.tell()
    file.seek(0)
    return size


@router.post("/stt")
def generate_stt(request: Request, audio: UploadFile = File(...)):
    try:
        if audio.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=415, detail="Only WAV audio files are supported"
            )

        file_size = _file_size(audio.file)

        if file_size == 0:
            raise HTTPException(status_code=400, detail="Audio file is empty")

        if file_size > AUDIO_MAX_SIZE:
            limit_mb = AUDIO_MAX_SIZE // (1024 * 1024)
            raise HTTPException(
                status_code=413, detail=f"Audio file exceeds {limit_mb} MB"
            )

        if not _is_wav(audio.file):
            raise HTTPException(
                status_code=415, detail="Only WAV audio files are supported"
            )

        result = transcribe_audio(request.app.state.stt_model, audio.file)

        if not result["text"]:
            raise HTTPException(status_code=422, detail="No speech detected")

        return {"service": "hivi-stt", "status": "success", "data": result}

    except HTTPException:
        raise

    except InvalidAudioError:
        raise HTTPException(status_code=400, detail="Audio file is invalid or corrupted")

    except AudioTooLongError:
        raise HTTPException(status_code=413, detail="Audio duration exceeds the limit")

    except Exception:
        logger.exception("STT generation error")
        raise HTTPException(status_code=500, detail="Failed to transcribe audio")

    finally:
        audio.file.close()