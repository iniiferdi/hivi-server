
import asyncio
import json
import os
from math import gcd

import numpy as np
import scipy.io.wavfile
import websockets
from dotenv import load_dotenv
from scipy.signal import resample_poly


load_dotenv()

TOKEN = os.getenv("HIVI_WS_TOKEN")

SERVER_URI = "ws://localhost:8000/ws/audio/desk-01"
INPUT_WAV = "test_hivi.wav"
OUTPUT_WAV = "hivi_response.wav"

TARGET_SAMPLE_RATE = 16000
CHUNK_SIZE = 4096

if not TOKEN:
    raise RuntimeError("HIVI_WS_TOKEN belum diatur di .env")


def prepare_audio(path):
    """Convert WAV to mono PCM16 16kHz."""

    sample_rate, audio = scipy.io.wavfile.read(path)

    if audio.ndim == 2:
        audio = audio.astype(np.float32).mean(axis=1)

    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        audio = audio.astype(np.float32) / max(abs(info.min), info.max)
    else:
        audio = audio.astype(np.float32)

    audio = np.nan_to_num(audio)
    audio = np.clip(audio, -1.0, 1.0)

    if sample_rate != TARGET_SAMPLE_RATE:
        divisor = gcd(sample_rate, TARGET_SAMPLE_RATE)

        audio = resample_poly(
            audio,
            TARGET_SAMPLE_RATE // divisor,
            sample_rate // divisor,
        )

    pcm = np.rint(audio * 32767).astype("<i2")

    return pcm.tobytes()


async def receive_turn(ws):
    """Receive PCM16 chunks and save the completed response as WAV."""

    audio_chunks = []
    sample_rate = None
    receiving_audio = False

    while True:
        message = await ws.recv()

        if isinstance(message, bytes):
            if not receiving_audio:
                print("Peringatan: menerima binary di luar sesi audio.")
                continue

            audio_chunks.append(message)
            continue

        data = json.loads(message)
        message_type = data.get("type")

        print("Server:", data)

        if message_type == "audio_start":
            if data.get("format") != "pcm_s16le":
                raise RuntimeError(
                    f"Format audio tidak didukung: {data.get('format')}"
                )

            sample_rate = data["sample_rate"]
            audio_chunks.clear()
            receiving_audio = True

        elif message_type == "audio_end":
            if receiving_audio:
                pcm_data = b"".join(audio_chunks)

                if len(pcm_data) % 2 != 0:
                    raise RuntimeError("Ukuran PCM16 tidak valid.")

                samples = np.frombuffer(pcm_data, dtype="<i2")

                scipy.io.wavfile.write(
                    OUTPUT_WAV,
                    sample_rate,
                    samples,
                )

                print(
                    f"Audio response saved to {OUTPUT_WAV} "
                    f"({len(pcm_data)} bytes, {sample_rate} Hz)."
                )

                receiving_audio = False

        elif message_type in ("turn_complete", "error"):
            return data


async def main():
    audio_bytes = prepare_audio(INPUT_WAV)

    async with websockets.connect(
        SERVER_URI,
        max_size=20 * 1024 * 1024,
        ping_interval=20,
        ping_timeout=20,
    ) as ws:

        await ws.send(json.dumps({
            "type": "auth",
            "token": TOKEN,
        }))

        print("Server:", await ws.recv())

        await ws.send(json.dumps({
            "type": "audio_start",
            "sample_rate": TARGET_SAMPLE_RATE,
            "format": "pcm_s16le",
        }))

        print("Server:", await ws.recv())

        for offset in range(0, len(audio_bytes), CHUNK_SIZE):
            await ws.send(
                audio_bytes[offset:offset + CHUNK_SIZE]
            )

        await ws.send(json.dumps({
            "type": "audio_end",
        }))

        result = await receive_turn(ws)

        if result.get("type") == "error":
            raise SystemExit(
                f"Turn gagal: {result.get('code')}"
            )


if __name__ == "__main__":
    asyncio.run(main())
