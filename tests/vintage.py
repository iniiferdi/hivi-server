import os
import sys
import subprocess
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import butter, sosfilt

os.makedirs("voices", exist_ok=True)

YOUTUBE_URL = "https://www.youtube.com/watch?v=JMRFHSypbOY"
START_SEC = "00:00:12"   # Musik latar berhenti di titik ini, vokal BMO bersih
DURATION_SEC = "9"       # Ambil 9 detik dialog bersih (sebelum bunyi klik akhir)

def build_bmo_reference():
    raw_audio = "voices/_temp_yt.m4a"
    trimmed_wav = "voices/_temp_24k.wav"
    out_wav = "voices/hivi_vintage.wav"

    # 1. Unduh stream audio terbaik dari YouTube memakai yt-dlp
    cmd_dl = [
        sys.executable, "-m", "yt_dlp",
        "-f", "bestaudio",
        "-o", raw_audio,
        YOUTUBE_URL,
    ]
    subprocess.run(cmd_dl, check=True)

    # 2. Potong bagian dialog bersih & konversi ke WAV 24kHz Mono
    cmd_ffmpeg = [
        "ffmpeg", "-y",
        "-ss", START_SEC,
        "-t", DURATION_SEC,
        "-i", raw_audio,
        "-ar", "24000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        trimmed_wav,
    ]
    subprocess.run(
        cmd_ffmpeg,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )

    sr, data = wavfile.read(trimmed_wav)
    audio = data.astype(np.float32) / 32768.0

    # 3. Bersihkan frekuensi sub-bass & desis berlebih tanpa merusak artikulasi vokal
    sos_bp = butter(3, [180, 7500], btype="bandpass", fs=sr, output="sos")
    audio = sosfilt(sos_bp, audio)

    # 4. Trim hening awal/akhir & normalisasi ke -3 dBFS
    non_silent = np.where(np.abs(audio) > 0.015)[0]
    if len(non_silent) > 0:
        start = max(0, non_silent[0] - int(0.12 * sr))
        end = min(len(audio), non_silent[-1] + int(0.15 * sr))
        audio = audio[start:end]

    peak = np.max(np.abs(audio)) + 1e-8
    audio = (audio / peak) * 0.72

    wavfile.write(out_wav, sr, (audio * 32767.0).astype(np.int16))

    # Bersihkan file sementara
    for temp_file in (raw_audio, trimmed_wav):
        if os.path.exists(temp_file):
            os.remove(temp_file)

    print(f"Selesai! Referensi suara BMO tersimpan di: {out_wav} ({len(audio)/sr:.2f} detik)")

if __name__ == "__main__":
    build_bmo_reference()