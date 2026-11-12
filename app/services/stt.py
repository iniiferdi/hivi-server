
from threading import Lock

from faster_whisper import WhisperModel


stt_lock = Lock()


def load_stt_model():
    print("Loading Faster-Whisper STT...")

    model = WhisperModel(
        "small",
        device="cpu",
        compute_type="int8",
        cpu_threads=4,
        num_workers=1
    )

    print("Hivi STT is ready!")

    return model


def transcribe_audio(model, audio_file):
    audio_file.seek(0)

    with stt_lock:
        segments, info = model.transcribe(
            audio_file,
            language="id",
            beam_size=3,
            vad_filter=False
        )

        # Transkripsi bersifat lazy, jadi harus dikonsumsi
        # sebelum lock dilepaskan.
        transcription = " ".join(
            segment.text.strip()
            for segment in segments
        ).strip()

    return {
        "text": transcription,
        "language": info.language,
        "duration": round(info.duration, 2)
    }
