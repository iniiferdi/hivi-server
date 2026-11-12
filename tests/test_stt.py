
from faster_whisper import WhisperModel
from pathlib import Path
import time

AUDIO_PATH = Path("test_hivi.wav")

if not AUDIO_PATH.exists():
    raise FileNotFoundError(f"Audio tidak ditemukan: {AUDIO_PATH}")

print("Loading Faster-Whisper...")
start_time = time.time()

model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8",
    cpu_threads=4,
    num_workers=1
)

print(f"Model berhasil dimuat dalam {time.time() - start_time:.2f} detik")

print("\nMemulai transkripsi...")
start_time = time.time()

segments, info = model.transcribe(
    str(AUDIO_PATH),
    language="id",
    beam_size=3,
    vad_filter=False
)

results = []

for segment in segments:
    results.append(segment.text)

transcription = "".join(results).strip()

print("\n===== HASIL STT HIVI =====")
print(f"Bahasa     : {info.language}")
print(f"Durasi     : {info.duration:.2f} detik")
print(f"Transkripsi: {transcription}")
print(f"Waktu STT  : {time.time() - start_time:.2f} detik")
