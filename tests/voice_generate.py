import asyncio
import os
import random
import subprocess
import edge_tts
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import butter, sosfilt

OUT_DIR = "dataset_hey_hivi/hey_hivi"
TEMP_DIR = "dataset_hey_hivi/_tmp"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

TARGET_SR = 16000
TARGET_DURATION = 1.5  # 1.5 detik sesuai window target KWS
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)

TEXT_VARIANTS = [
    "Hey hivi",
    "Hei hivi",
    "Hey, hee-vee",
    "Hei, hi-vi",
]

VOICES = [
    "id-ID-GadisNeural",
    "id-ID-ArdiNeural",
    "en-US-JennyNeural",
    "en-US-GuyNeural",
    "en-US-AriaNeural",
    "en-GB-SoniaNeural",
    "en-AU-NatashaNeural",
]

PITCHES = ["-8Hz", "-4Hz", "+0Hz", "+5Hz", "+10Hz"]
RATES = ["-10%", "-3%", "+0%", "+6%", "+12%"]


def augment_for_inmp441(audio: np.ndarray) -> np.ndarray:
    """Menyesuaikan audio TTS agar mendekati karakteristik mic INMP441 di ruangan."""
    active = np.where(np.abs(audio) > 0.02)[0]
    if len(active) > 0:
        pad = int(0.05 * TARGET_SR)
        start = max(0, active[0] - pad)
        end = min(len(audio), active[-1] + pad)
        audio = audio[start:end]

    max_speech_len = int(1.35 * TARGET_SR)
    if len(audio) > max_speech_len:
        indices = np.linspace(0, len(audio) - 1, max_speech_len)
        audio = np.interp(indices, np.arange(len(audio)), audio)

    canvas = np.zeros(TARGET_SAMPLES, dtype=np.float32)
    slack = TARGET_SAMPLES - len(audio)
    offset = random.randint(0, max(0, slack))
    canvas[offset : offset + len(audio)] = audio

    sos = butter(2, [100, 7200], btype="bandpass", fs=TARGET_SR, output="sos")
    canvas = sosfilt(sos, canvas)

    peak = np.max(np.abs(canvas)) + 1e-8
    target_gain = random.uniform(0.35, 0.95)
    canvas = (canvas / peak) * target_gain

    noise_level = random.uniform(0.001, 0.008)
    room_noise = np.random.normal(0, noise_level, TARGET_SAMPLES).astype(np.float32)
    canvas = np.clip(canvas + room_noise, -1.0, 1.0)

    return canvas


async def generate_one(idx: int, max_retries: int = 3):
    out_path = os.path.join(OUT_DIR, f"ai_hey_hivi_{idx:04d}.wav")

    # Skip jika file sudah pernah dibuat (aman kalau skrip diulang)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        return

    mp3_path = os.path.join(TEMP_DIR, f"raw_{idx}.mp3")
    wav_path = os.path.join(TEMP_DIR, f"raw_{idx}.wav")

    for attempt in range(1, max_retries + 1):
        voice = random.choice(VOICES)
        text = random.choice(TEXT_VARIANTS)
        pitch = random.choice(PITCHES)
        rate = random.choice(RATES)

        try:
            # Jeda acak singkat agar tidak kena rate-limit server TTS
            await asyncio.sleep(random.uniform(0.15, 0.55))

            communicate = edge_tts.Communicate(
                text=text,
                voice=voice,
                pitch=pitch,
                rate=rate,
            )
            await communicate.save(mp3_path)

            # Pastikan file MP3 benar-benar terunduh dan tidak kosong
            if not os.path.exists(mp3_path) or os.path.getsize(mp3_path) < 256:
                raise RuntimeError("Output MP3 kosong dari server TTS")

            subprocess.run(
                ["ffmpeg", "-y", "-i", mp3_path, "-ar", str(TARGET_SR), "-ac", "1", wav_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )

            _, data = wavfile.read(wav_path)
            if len(data) == 0:
                raise RuntimeError("Hasil konversi WAV kosong")

            audio_f32 = data.astype(np.float32) / 32768.0
            aug_audio = augment_for_inmp441(audio_f32)
            wavfile.write(out_path, TARGET_SR, (aug_audio * 32767.0).astype(np.int16))
            return

        except Exception as e:
            if attempt == max_retries:
                print(f"[SKIP] Sampel #{idx} gagal setelah {max_retries}x percobaan: {e}")
            else:
                await asyncio.sleep(1.5 * attempt)
        finally:
            for tmp in (mp3_path, wav_path):
                if os.path.exists(tmp):
                    os.remove(tmp)


async def main(total_samples: int = 300, concurrency: int = 2):
    sem = asyncio.Semaphore(concurrency)
    completed = 0

    async def worker(i):
        nonlocal completed
        async with sem:
            await generate_one(i)
            completed += 1
            if completed % 25 == 0 or completed == total_samples:
                print(f"Progress: {completed}/{total_samples} file diproses...")

    await asyncio.gather(*(worker(i) for i in range(total_samples)))
    total_valid = len([f for f in os.listdir(OUT_DIR) if f.endswith(".wav")])
    print(f"\nSelesai! Total {total_valid} file WAV siap pakai di: {OUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main(total_samples=300, concurrency=2))