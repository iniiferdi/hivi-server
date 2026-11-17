import asyncio
import os
import random
import subprocess
import edge_tts
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import butter, sosfilt

OUT_DIR = "dataset_hey_hivi/unknown"
TEMP_DIR = "dataset_hey_hivi/_tmp"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

TARGET_SR = 16000
TARGET_DURATION = 1.5
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)

# 1. Kata jebakan (Hard Negatives): potongan wake-word & kata mirip bunyi "hey hivi"
HARD_NEGATIVES = [
    "Hey", "Hei", "Hivi", "Hee-vee", "Hi-vi", "Hifi", "WiFi",
    "Hello hivi", "Halo hivi", "Hai hivi", "Oke hivi", "Ya hivi",
    "Heavy", "TV", "Televisi", "Kiwi", "Vivi", "Stevie", "Ivy", "Evie", "Levi",
    "Navy", "Baby", "Maybe", "Crazy", "Lazy", "Hay day", "Hey yo",
    "Hey bro", "Hey guys", "Hei kamu", "Hei sini", "Hei lihat",
    "Hey Siri", "Hey Google", "Ok Google", "Hi there", "Hey there",
]

# 2. Percakapan sehari-hari & perintah umum Bahasa Indonesia
INDONESIAN_PHRASES = [
    "Halo apa kabar", "Selamat pagi", "Selamat siang", "Selamat sore", "Selamat malam",
    "Terima kasih banyak", "Sama-sama", "Maaf ya", "Permisi", "Tolong bantu saya",
    "Jam berapa sekarang", "Tanggal berapa hari ini", "Cuaca hari ini cerah",
    "Di luar hujan deras", "Hari ini panas sekali", "Malam ini dingin banget",
    "Nyalakan lampu meja", "Matikan musiknya", "Kecilkan suaranya", "Besarkan volume",
    "Oke siap laksanakan", "Iya betul sekali", "Tidak tahu", "Bukan yang itu",
    "Sebentar ya", "Tunggu sebentar", "Lagi ngapain sekarang", "Sudah selesai belum",
    "Mantap sekali", "Keren banget", "Bagus sekali", "Tidak masalah",
    "Jangan lupa istirahat", "Besok kita berangkat", "Kemarin sore", "Nanti malam",
    "Siapa namamu", "Dari mana asalnya", "Mau ke mana", "Gimana kabarnya",
    "Alhamdulillah baik", "Coba ulangi lagi", "Mulai sekarang", "Berhenti dulu",
    "Lanjutkan saja", "Simpan data", "Hapus file", "Kirim pesan",
    "Baca berita terbaru", "Putar lagu santai", "Cek koneksi internet",
    "Satu dua tiga", "Empat lima enam", "Tujuh delapan sembilan", "Sepuluh",
    "Buka pintu", "Tutup jendela", "Mau makan apa", "Sudah makan belum",
    "Semangat kerja", "Hati-hati di jalan", "Sampai jumpa besok",
]

# 3. Kosakata & perintah singkat Bahasa Inggris
ENGLISH_PHRASES = [
    "Hello", "Hi", "Good morning", "Good afternoon", "Good evening", "Good night",
    "How are you", "What time is it", "What is the weather", "Set a timer",
    "Play some music", "Stop the music", "Volume up", "Volume down", "Turn off",
    "System ready", "Connecting now", "Thank you very much", "You are welcome",
    "Excuse me", "Sorry about that", "Yes please", "No thanks", "Maybe later",
    "See you tomorrow", "Have a nice day", "Check the status", "Restart system",
    "Testing microphone", "Computer", "Assistant", "One two three", "Four five six",
]

# 13 suara lintas aksen (Indonesia, Melayu, US, UK, AU, CA, IN)
VOICES = [
    "id-ID-GadisNeural",
    "id-ID-ArdiNeural",
    "ms-MY-YasminNeural",
    "ms-MY-OsmanNeural",
    "en-US-JennyNeural",
    "en-US-GuyNeural",
    "en-US-AriaNeural",
    "en-GB-SoniaNeural",
    "en-GB-RyanNeural",
    "en-AU-NatashaNeural",
    "en-CA-ClaraNeural",
    "en-IN-NeerjaNeural",
    "en-IN-PrabhatNeural",
]

PITCHES = ["-10Hz", "-5Hz", "+0Hz", "+5Hz", "+10Hz"]
RATES = ["-12%", "-4%", "+0%", "+6%", "+14%"]


def create_augmented_variant(audio: np.ndarray) -> np.ndarray:
    """Membuat 1 variasi akustik unik (speed stretch, posisi waktu, jarak mic, dan noise)."""
    active = np.where(np.abs(audio) > 0.02)[0]
    if len(active) > 0:
        pad = int(0.05 * TARGET_SR)
        start = max(0, active[0] - pad)
        end = min(len(audio), active[-1] + pad)
        audio = audio[start:end]

    # Variasi tempo halus (0.90x - 1.10x)
    stretch = random.uniform(0.90, 1.10)
    new_len = max(1, int(len(audio) * stretch))
    max_speech_len = int((TARGET_DURATION - 0.12) * TARGET_SR)
    new_len = min(new_len, max_speech_len)

    indices = np.linspace(0, len(audio) - 1, new_len)
    audio = np.interp(indices, np.arange(len(audio)), audio)

    # Geser posisi ucapan secara acak dalam window 1.5 detik
    canvas = np.zeros(TARGET_SAMPLES, dtype=np.float32)
    slack = TARGET_SAMPLES - len(audio)
    offset = random.randint(0, max(0, slack))
    canvas[offset : offset + len(audio)] = audio

    # Simulasi gema ruangan tipis secara acak (30% peluang)
    if random.random() < 0.30:
        delay = random.randint(int(0.015 * TARGET_SR), int(0.035 * TARGET_SR))
        echo = np.pad(canvas, (delay, 0))[:-delay]
        canvas = canvas + random.uniform(0.10, 0.22) * echo

    # Karakter frekuensi mic INMP441
    sos = butter(2, [100, 7200], btype="bandpass", fs=TARGET_SR, output="sos")
    canvas = sosfilt(sos, canvas)

    # Variasi jarak bicara (volume 25% - 96%)
    peak = np.max(np.abs(canvas)) + 1e-8
    canvas = (canvas / peak) * random.uniform(0.25, 0.96)

    # Noise ruangan acak
    noise_level = random.uniform(0.0008, 0.009)
    room_noise = np.random.normal(0, noise_level, TARGET_SAMPLES).astype(np.float32)
    return np.clip(canvas + room_noise, -1.0, 1.0)


async def generate_batch(batch_idx: int, variants_per_tts: int = 3, max_retries: int = 3):
    start_id = batch_idx * variants_per_tts
    out_paths = [
        os.path.join(OUT_DIR, f"ai_unknown_{start_id + v:04d}.wav")
        for v in range(variants_per_tts)
    ]

    # Skip batch jika seluruh file variannya sudah ada
    if all(os.path.exists(p) and os.path.getsize(p) > 1000 for p in out_paths):
        return

    mp3_path = os.path.join(TEMP_DIR, f"unk_batch_{batch_idx}.mp3")
    wav_path = os.path.join(TEMP_DIR, f"unk_batch_{batch_idx}.wav")

    for attempt in range(1, max_retries + 1):
        voice = random.choice(VOICES)
        # Komposisi: 40% kata jebakan, 40% frasa Indonesia, 20% frasa Inggris
        r = random.random()
        if r < 0.40:
            text = random.choice(HARD_NEGATIVES)
        elif r < 0.80:
            text = random.choice(INDONESIAN_PHRASES)
        else:
            text = random.choice(ENGLISH_PHRASES)

        pitch = random.choice(PITCHES)
        rate = random.choice(RATES)

        try:
            await asyncio.sleep(random.uniform(0.15, 0.50))

            communicate = edge_tts.Communicate(
                text=text,
                voice=voice,
                pitch=pitch,
                rate=rate,
            )
            await communicate.save(mp3_path)

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

            # Hasilkan 3 variasi akustik berbeda dari 1 rekaman TTS
            for out_p in out_paths:
                aug_audio = create_augmented_variant(audio_f32)
                wavfile.write(out_p, TARGET_SR, (aug_audio * 32767.0).astype(np.int16))
            return

        except Exception as e:
            if attempt == max_retries:
                print(f"[SKIP] Batch #{batch_idx} gagal setelah {max_retries}x percobaan: {e}")
            else:
                await asyncio.sleep(1.5 * attempt)
        finally:
            for tmp in (mp3_path, wav_path):
                if os.path.exists(tmp):
                    os.remove(tmp)


async def main(total_batches: int = 340, variants_per_tts: int = 3, concurrency: int = 2):
    sem = asyncio.Semaphore(concurrency)
    completed = 0

    async def worker(b):
        nonlocal completed
        async with sem:
            await generate_batch(b, variants_per_tts=variants_per_tts)
            completed += 1
            if completed % 20 == 0 or completed == total_batches:
                print(f"Progress: {completed * variants_per_tts}/{total_batches * variants_per_tts} file...")

    await asyncio.gather(*(worker(b) for b in range(total_batches)))
    total_valid = len([f for f in os.listdir(OUT_DIR) if f.endswith(".wav")])
    print(f"\nSelesai! Total {total_valid} file WAV unknown tersimpan di: {OUT_DIR}")


if __name__ == "__main__":
    # 340 batch x 3 variasi = 1.020 file WAV unknown
    asyncio.run(main(total_batches=340, variants_per_tts=3, concurrency=2))