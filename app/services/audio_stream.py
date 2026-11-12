"""WebSocket audio Hivi, protokol v2.

Client -> server
  {"type":"auth","token":...}
  {"type":"audio_start","proto":2,"utterance_id":ID,"format":"mulaw"|"pcm_s16le","sample_rate":16000}
  binary: [seq u32 LE][data]            (frame bernomor, mulai dari 0)
  {"type":"audio_end","utterance_id":ID,"bytes":N,"frames":F,"crc32":C}
  {"type":"resume","utterance_id":ID}   (setelah sambung ulang)
  {"type":"cancel"} / {"type":"ping"}

Server -> client
  connected, recording, ack{seq}, nack{next_seq}, resume_ok{state,next_seq},
  upload_ok, processing, transcript, response, audio_start, <binary PCM>,
  audio_end, turn_complete, error, cancelled, pong

Pemrosesan (STT -> Brain -> TTS) berjalan di task terpisah per ucapan, sehingga
handler tetap membaca pesan (cancel/ping) dan balasan dikirim ke koneksi AKTIF
perangkat, bukan koneksi lama yang mungkin sudah mati.
"""

import asyncio
import json
import os
import re
import secrets
import time
import traceback
import zlib
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import numpy as np
import scipy.io.wavfile

from fastapi import WebSocket
from fastapi.websockets import WebSocketDisconnect

from app.config import (
    HIVI_WS_TOKEN,
    WS_AUTH_TIMEOUT,
    WS_IDLE_TIMEOUT,
    WS_MAX_CHUNK_BYTES,
    AUDIO_SAMPLE_RATE,
    AUDIO_MAX_SIZE,
    AUDIO_MAX_DURATION,
)


PROTOCOL_VERSION = 2

DEVICE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
UTTERANCE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

FRAME_HEADER_BYTES = 4
BYTES_PER_SAMPLE = {"pcm_s16le": 2, "mulaw": 1}

# Upload
ACK_EVERY_FRAMES = 8          # ack setiap N frame...
ACK_MAX_INTERVAL = 0.5        # ...atau setiap sekian detik bila ada frame baru
NACK_MIN_INTERVAL = 0.5       # jarak minimal antar nack
UPLOAD_GAP_WARN = 1.0         # jeda antar frame yang dicatat (detik)
TURN_TTL = 120.0              # ucapan yang tidak aktif dibuang setelah ini (detik)

# Pemrosesan
SLOT_WAIT = 20.0              # tunggu giliran pemrosesan (detik)
RESULT_WAIT = 30.0            # tunggu perangkat tersambung untuk menerima hasil (detik)
EVICT_WAIT = 3.0              # tunggu koneksi lama dibersihkan saat diusir (detik)

# Total waktu maksimal untuk seluruh proses AI Brain, termasuk retry.
BRAIN_TIMEOUT = 90
BRAIN_ATTEMPT_TIMEOUT = 45
BRAIN_MAX_ATTEMPTS = 2
BRAIN_RETRY_BASE_DELAY = 1.0

TRANSIENT_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
TRANSIENT_ERROR_NAMES = frozenset({
    "ServerError",
    "ConnectError",
    "ConnectTimeout",
    "ReadTimeout",
    "ReadError",
    "RemoteProtocolError",
})

AUDIO_OUTPUT_CHUNK_BYTES = 4096

# Simpan audio per turn (sudah di-decode ke PCM) untuk debugging.
DEBUG_SAVE_AUDIO = os.getenv("HIVI_DEBUG_AUDIO", "1") == "1"
DEBUG_KEEP_FILES = 30
DEBUG_DIR = Path(__file__).resolve().parents[2] / "data" / "debug_audio"

# Riwayat percakapan: "device" = berlanjut antar koneksi/reboot (disarankan),
# "connection" = riwayat baru di setiap koneksi.
HISTORY_SCOPE = os.getenv("HIVI_HISTORY_SCOPE", "device")

# Bila AI Brain gagal, robot tetap berbicara (bukan diam).
SPOKEN_FALLBACK = os.getenv("HIVI_SPOKEN_FALLBACK", "1") == "1"
FALLBACK_TEXT = os.getenv(
    "HIVI_FALLBACK_TEXT",
    "Maaf, aku sedang kesulitan berpikir. Coba tanya lagi sebentar lagi ya.",
)


# ---------------------------------------------------------------- Registri

class DeviceSession:
    """Satu koneksi WebSocket aktif milik satu device_id."""

    def __init__(self, websocket):
        self.websocket = websocket
        self.session_id = str(uuid4())
        self.evict = asyncio.Event()   # diset bila koneksi ini harus diusir
        self.done = asyncio.Event()    # diset saat handler koneksi ini selesai
        self.dead = False              # pengiriman pernah gagal
        self._lock = asyncio.Lock()    # serialisasi kirim dari handler dan task turn

    async def send_json(self, payload):
        async with self._lock:
            await self.websocket.send_json(payload)

    async def send_bytes(self, data):
        async with self._lock:
            await self.websocket.send_bytes(data)


class Evicted(Exception):
    """Koneksi digantikan oleh koneksi baru dari perangkat yang sama."""


class Turn:
    """Satu ucapan: dari upload sampai balasan."""

    def __init__(self, device_id, utterance_id, fmt, history_id):
        self.device_id = device_id
        self.utterance_id = utterance_id
        self.fmt = fmt
        self.bps = BYTES_PER_SAMPLE[fmt]
        self.history_id = history_id

        self.data = bytearray()
        self.next_seq = 0
        self.acked_seq = 0
        self.last_ack_at = time.monotonic()
        self.last_nack_at = 0.0

        self.state = "uploading"       # uploading | processing | done | failed
        self.task = None

        self.created = time.monotonic()
        self.updated = self.created
        self.last_frame_at = self.created
        self.max_gap = 0.0
        self.frames = 0


_sessions: dict[str, DeviceSession] = {}
_turns: dict[str, Turn] = {}

# Cache audio fallback (teks -> (pcm16, sample_rate)).
_fallback_cache: dict[str, tuple[bytes, int]] = {}


def prune_turns():
    now = time.monotonic()

    for device_id, turn in list(_turns.items()):
        if turn.state != "processing" and now - turn.updated > TURN_TTL:
            _turns.pop(device_id, None)


def drop_turn_if_same(device_id, turn):
    if _turns.get(device_id) is turn:
        _turns.pop(device_id, None)


async def wait_session(device_id, wait):
    """Koneksi aktif perangkat; menunggu hingga 'wait' detik bila belum ada."""

    deadline = time.monotonic() + wait

    while True:
        session = _sessions.get(device_id)

        if session is not None and not session.dead and not session.evict.is_set():
            return session

        if time.monotonic() >= deadline:
            return None

        await asyncio.sleep(0.25)


async def notify(device_id, payload):
    """Kirim pesan informatif ke koneksi aktif; kegagalan diabaikan."""

    session = _sessions.get(device_id)

    if session is None or session.dead:
        return

    try:
        await session.send_json(payload)
    except Exception:
        session.dead = True


# ---------------------------------------------------------------- Audio

def mulaw_decode(data: bytes) -> np.ndarray:
    """G.711 mu-law -> PCM16."""

    u = (~np.frombuffer(data, dtype=np.uint8)).astype(np.int32) & 0xFF

    sign = u & 0x80
    exponent = (u >> 4) & 0x07
    mantissa = u & 0x0F

    magnitude = (((mantissa << 3) + 0x84) << exponent) - 0x84
    sample = np.where(sign != 0, -magnitude, magnitude)

    return sample.astype("<i2")


def decode_audio(fmt, data) -> np.ndarray:
    if fmt == "pcm_s16le":
        return np.frombuffer(bytes(data), dtype="<i2")

    if fmt == "mulaw":
        return mulaw_decode(bytes(data))

    raise ValueError(f"Format tidak didukung: {fmt}")


def generate_pcm16(model, voice_state, text, lock):
    """Generate mono PCM16 audio using Pocket TTS."""

    with lock:
        audio = model.generate_audio(voice_state, text)

    audio_np = audio.detach().cpu().numpy()
    audio_np = np.asarray(audio_np, dtype=np.float32).squeeze()

    if audio_np.size == 0:
        raise RuntimeError("TTS menghasilkan audio kosong.")

    audio_np = np.nan_to_num(
        audio_np,
        nan=0.0,
        posinf=1.0,
        neginf=-1.0,
    )

    audio_np = np.clip(audio_np, -1.0, 1.0)

    pcm16 = np.rint(audio_np * 32767.0).astype("<i2")

    return pcm16.tobytes()


def save_debug_audio(samples, name):
    """Tulis WAV debug dan pangkas file lama. Dijalankan di thread."""

    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    path = DEBUG_DIR / f"{name}.wav"

    scipy.io.wavfile.write(str(path), AUDIO_SAMPLE_RATE, samples)

    files = sorted(DEBUG_DIR.glob("*.wav"), key=lambda p: p.stat().st_mtime)
    for old in files[:-DEBUG_KEEP_FILES]:
        try:
            old.unlink()
        except OSError:
            pass

    return path


# ---------------------------------------------------------------- Brain

def is_transient_error(error):
    code = getattr(error, "status_code", None) or getattr(error, "code", None)

    if code in TRANSIENT_STATUS_CODES:
        return True

    if isinstance(error, (ConnectionError, TimeoutError)):
        return True

    return type(error).__name__ in TRANSIENT_ERROR_NAMES


def describe_error(error):
    code = getattr(error, "code", None)
    status = getattr(error, "status", None)
    message = getattr(error, "message", None) or str(error)

    return (
        f"{type(error).__name__} code={code} status={status} "
        f"message={message[:500]}"
    )


async def request_brain_response(app, message, history):
    """Request a Gemini response with retry for transient failures."""

    deadline = time.monotonic() + BRAIN_TIMEOUT
    last_error = None

    for attempt in range(1, BRAIN_MAX_ATTEMPTS + 1):
        remaining = deadline - time.monotonic()

        if remaining <= 0:
            break

        try:
            return await asyncio.wait_for(
                app.state.generate_response(
                    message=message,
                    history=history,
                ),
                timeout=min(BRAIN_ATTEMPT_TIMEOUT, remaining),
            )

        except asyncio.TimeoutError as error:
            last_error = error
            print(f"[BRAIN] Attempt {attempt}/{BRAIN_MAX_ATTEMPTS} timed out.")

        except Exception as error:
            if not is_transient_error(error):
                raise

            last_error = error
            print(
                f"[BRAIN] Attempt {attempt}/{BRAIN_MAX_ATTEMPTS} "
                f"failed with transient error: {describe_error(error)}"
            )

        if attempt < BRAIN_MAX_ATTEMPTS:
            delay = BRAIN_RETRY_BASE_DELAY * (2 ** (attempt - 1))

            if deadline - time.monotonic() <= delay:
                break

            await asyncio.sleep(delay)

    if last_error is None or isinstance(last_error, asyncio.TimeoutError):
        raise asyncio.TimeoutError

    raise last_error


# ---------------------------------------------------------------- Balasan

async def reply_error(turn, message, code):
    """Kirim error turn ke koneksi aktif (menunggu bila sedang tersambung ulang)."""

    session = await wait_session(turn.device_id, RESULT_WAIT)

    if session is None:
        print(f"[WS] {turn.device_id}: perangkat tidak tersambung, error {code} dibuang")
        return False

    try:
        await session.send_json({"type": "error", "code": code, "message": message})
        return True
    except Exception:
        session.dead = True
        return False


async def reply_audio(turn, audio_output, sample_rate):
    """Stream audio balasan ke koneksi aktif. True bila terkirim penuh."""

    session = await wait_session(turn.device_id, RESULT_WAIT)

    if session is None:
        print(f"[WS] {turn.device_id}: perangkat tidak tersambung, balasan dibuang")
        return False

    chunk_size = AUDIO_OUTPUT_CHUNK_BYTES

    try:
        await session.send_json({
            "type": "audio_start",
            "format": "pcm_s16le",
            "sample_rate": sample_rate,
            "channels": 1,
            "encoding": "pcm_s16le",
            "size": len(audio_output),
            "chunk_size": chunk_size,
        })

        for offset in range(0, len(audio_output), chunk_size):
            await session.send_bytes(audio_output[offset:offset + chunk_size])

        await session.send_json({"type": "audio_end"})
        await session.send_json({"type": "turn_complete"})
        return True

    except asyncio.CancelledError:
        raise

    except Exception:
        session.dead = True
        print(f"[WS] {turn.device_id}: koneksi putus saat mengirim balasan")
        return False


async def get_fallback_audio(app):
    cached = _fallback_cache.get(FALLBACK_TEXT)

    if cached is None:
        audio = await asyncio.to_thread(
            generate_pcm16,
            app.state.tts_model,
            app.state.voice_state,
            FALLBACK_TEXT,
            app.state.tts_lock,
        )
        cached = (audio, app.state.tts_model.sample_rate)
        _fallback_cache[FALLBACK_TEXT] = cached

    return cached


async def speak_fallback(app, turn, reason):
    """Robot tetap berbicara saat AI Brain gagal. True bila dikirim."""

    if not SPOKEN_FALLBACK:
        return False

    try:
        audio, sample_rate = await get_fallback_audio(app)
    except Exception:
        print(f"[FALLBACK] Gagal membuat audio cadangan:\n{traceback.format_exc()}")
        return False

    print(f"[FALLBACK] Mengucapkan pesan cadangan ({reason})")

    await notify(turn.device_id, {
        "type": "response",
        "text": FALLBACK_TEXT,
        "degraded": True,
        "reason": reason,
    })
    await notify(turn.device_id, {"type": "processing", "stage": "tts"})
    await reply_audio(turn, audio, sample_rate)

    return True


# ---------------------------------------------------------------- Pipeline

async def process_turn(app, turn):
    """STT -> Gemini -> TTS untuk satu ucapan yang sudah lengkap."""

    device_id = turn.device_id

    try:
        samples = decode_audio(turn.fmt, turn.data)
    except Exception:
        print(f"[AUDIO] Gagal decode:\n{traceback.format_exc()}")
        await reply_error(turn, "Audio tidak valid.", "INVALID_AUDIO")
        return

    if samples.size == 0:
        await reply_error(turn, "Audio yang diterima kosong.", "EMPTY_AUDIO")
        return

    turn_name = f"{turn.utterance_id[:12]}"
    duration = samples.size / AUDIO_SAMPLE_RATE
    peak = int(np.abs(samples.astype(np.int32)).max())

    print(
        f"[AUDIO] {device_id}/{turn_name} | {turn.fmt} | Bytes: {len(turn.data)} | "
        f"Samples: {samples.size} | Duration: {duration:.2f}s | Peak: {peak}"
    )

    if DEBUG_SAVE_AUDIO:
        try:
            await asyncio.to_thread(save_debug_audio, samples, f"{device_id}-{turn_name}")
        except Exception as error:
            print(f"[AUDIO] Gagal menyimpan debug audio: {error!r}")

    wav_buffer = BytesIO()
    scipy.io.wavfile.write(wav_buffer, AUDIO_SAMPLE_RATE, samples)
    wav_buffer.seek(0)

    # ---------------- STT ----------------
    await notify(device_id, {"type": "processing", "stage": "stt"})
    stt_started = time.monotonic()

    try:
        stt_result = await asyncio.to_thread(
            app.state.stt_transcribe,
            app.state.stt_model,
            wav_buffer,
        )
        transcript = stt_result.get("text", "").strip()

    except Exception:
        print(f"[STT] Error:\n{traceback.format_exc()}")
        await reply_error(turn, "Gagal mengenali suara.", "STT_ERROR")
        return

    if not transcript:
        await reply_error(turn, "Tidak ada suara yang terdeteksi.", "NO_SPEECH")
        return

    print(f"[STT] ({time.monotonic() - stt_started:.2f}s) Transcript: {transcript!r}")
    await notify(device_id, {"type": "transcript", "text": transcript})

    # ---------------- Brain ----------------
    await notify(device_id, {"type": "processing", "stage": "brain"})

    try:
        history = await app.state.get_history(turn.history_id)
    except Exception:
        print(f"[BRAIN] get_history error (lanjut tanpa riwayat):\n{traceback.format_exc()}")
        history = []

    print("[BRAIN] Sending request to Gemini...")
    brain_started = time.monotonic()

    try:
        answer = await request_brain_response(app, transcript, history)

    except asyncio.TimeoutError:
        print("[BRAIN] Gemini request timed out.")

        if await speak_fallback(app, turn, "BRAIN_TIMEOUT"):
            return

        await reply_error(turn, "AI Brain membutuhkan waktu terlalu lama.", "BRAIN_TIMEOUT")
        return

    except Exception as error:
        print(
            f"[BRAIN] Gemini error: {describe_error(error)}\n"
            f"{traceback.format_exc()}"
        )

        if await speak_fallback(app, turn, "BRAIN_ERROR"):
            return

        await reply_error(turn, "Gagal mendapatkan respons dari AI Brain.", "BRAIN_ERROR")
        return

    if not isinstance(answer, str) or not answer.strip():
        print(f"[BRAIN] Respons kosong/tidak valid: {answer!r}")

        if await speak_fallback(app, turn, "BRAIN_EMPTY"):
            return

        await reply_error(turn, "AI Brain mengembalikan respons kosong.", "BRAIN_ERROR")
        return

    answer = answer.strip()

    print(f"[BRAIN] Response received in {time.monotonic() - brain_started:.2f}s: {answer}")

    try:
        await app.state.save_exchange(
            session_id=turn.history_id,
            user_message=transcript,
            assistant_message=answer,
        )
    except Exception:
        print(f"[BRAIN] save_exchange error:\n{traceback.format_exc()}")

    await notify(device_id, {"type": "response", "text": answer})

    # ---------------- TTS ----------------
    await notify(device_id, {"type": "processing", "stage": "tts"})
    tts_started = time.monotonic()

    try:
        audio_output = await asyncio.to_thread(
            generate_pcm16,
            app.state.tts_model,
            app.state.voice_state,
            answer,
            app.state.tts_lock,
        )
        sample_rate = app.state.tts_model.sample_rate

    except Exception:
        print(f"[TTS] Error:\n{traceback.format_exc()}")
        await reply_error(turn, "Gagal membuat suara balasan.", "TTS_ERROR")
        return

    print(f"[TTS] ({time.monotonic() - tts_started:.2f}s) {len(audio_output)} byte")

    await reply_audio(turn, audio_output, sample_rate)


async def run_turn(app, turn):
    """Task pemrosesan satu ucapan; terlepas dari koneksi yang mengunggahnya."""

    acquired = False

    try:
        try:
            await asyncio.wait_for(
                app.state.audio_processing_slots.acquire(),
                timeout=SLOT_WAIT,
            )
            acquired = True
        except asyncio.TimeoutError:
            await reply_error(turn, "Server sedang memproses audio lain.", "SERVER_BUSY")
            return

        await process_turn(app, turn)

    except asyncio.CancelledError:
        print(f"[WS] {turn.device_id}: pemrosesan dibatalkan")
        raise

    except Exception:
        print(f"[WS] {turn.device_id}: error pemrosesan:\n{traceback.format_exc()}")
        await reply_error(turn, "Terjadi kesalahan pada server.", "INTERNAL_ERROR")

    finally:
        if acquired:
            app.state.audio_processing_slots.release()

        turn.state = "done"
        turn.updated = time.monotonic()
        turn.data = bytearray()          # bebaskan memori

        asyncio.get_running_loop().call_later(
            TURN_TTL,
            drop_turn_if_same,
            turn.device_id,
            turn,
        )


# ---------------------------------------------------------------- Penerimaan

async def send_error(session, message, code):
    """Kirim error protokol lalu tutup koneksi."""

    try:
        await session.send_json({"type": "error", "message": message, "code": code})
        await session.websocket.close(code=1008)
    except Exception:
        pass


async def receive_message(websocket, session):
    """Terima pesan berikutnya; berhenti bila idle terlalu lama atau diusir."""

    recv = asyncio.ensure_future(websocket.receive())
    evict = asyncio.ensure_future(session.evict.wait())

    try:
        done, _ = await asyncio.wait(
            {recv, evict},
            timeout=WS_IDLE_TIMEOUT,
            return_when=asyncio.FIRST_COMPLETED,
        )
    except BaseException:
        recv.cancel()
        evict.cancel()
        raise

    if recv in done:
        evict.cancel()
        return recv.result()

    recv.cancel()
    evict.cancel()

    if evict in done:
        raise Evicted

    raise asyncio.TimeoutError


async def authenticate(websocket):
    """Authenticate the connecting device."""

    if not HIVI_WS_TOKEN:
        await websocket.close(code=1011)
        return False

    try:
        message = await asyncio.wait_for(
            websocket.receive_json(),
            timeout=WS_AUTH_TIMEOUT,
        )
    except asyncio.TimeoutError:
        await websocket.close(code=1008)
        return False
    except (ValueError, WebSocketDisconnect):
        return False

    if message.get("type") != "auth":
        await websocket.close(code=1008)
        return False

    token = message.get("token", "")

    if not isinstance(token, str):
        await websocket.close(code=1008)
        return False

    if not secrets.compare_digest(token, HIVI_WS_TOKEN):
        await websocket.close(code=1008)
        return False

    return True


async def fail_turn(session, turn, code, message):
    """Gagalkan ucapan yang sedang diunggah (koneksi tetap hidup)."""

    print(f"[WS] {turn.device_id}: ucapan gagal ({code})")

    turn.state = "failed"
    turn.updated = time.monotonic()
    turn.data = bytearray()

    await session.send_json({"type": "error", "code": code, "message": message})


async def handle_audio_start(session, device_id, data):
    uid = data.get("utterance_id")
    fmt = data.get("format")

    if not isinstance(uid, str) or not UTTERANCE_ID_PATTERN.fullmatch(uid):
        await send_error(session, "utterance_id tidak valid.", "INVALID_UTTERANCE")
        return False

    if data.get("sample_rate") != AUDIO_SAMPLE_RATE:
        await send_error(session, "Sample rate harus 16000 Hz.", "INVALID_SAMPLE_RATE")
        return False

    if fmt not in BYTES_PER_SAMPLE:
        await send_error(session, "Format harus mulaw atau pcm_s16le.", "INVALID_FORMAT")
        return False

    prune_turns()

    # Ucapan baru menggantikan yang lama; pemrosesan lama dibatalkan.
    old = _turns.get(device_id)
    if old is not None and old.task is not None and not old.task.done():
        print(f"[WS] {device_id}: ucapan baru membatalkan pemrosesan sebelumnya")
        old.task.cancel()

    history_id = device_id if HISTORY_SCOPE == "device" else session.session_id
    _turns[device_id] = Turn(device_id, uid, fmt, history_id)

    await session.send_json({
        "type": "recording",
        "status": "started",
        "utterance_id": uid,
    })
    return True


async def handle_resume(session, device_id, data):
    uid = data.get("utterance_id")
    turn = _turns.get(device_id)

    state = "unknown"
    next_seq = 0

    if turn is not None and isinstance(uid, str) and turn.utterance_id == uid:
        if turn.state == "uploading":
            state = "uploading"
            next_seq = turn.next_seq
            turn.updated = time.monotonic()
            turn.last_ack_at = turn.updated
        elif turn.state == "processing":
            state = "processing"

    print(f"[WS] {device_id}: resume {uid} -> {state} (next_seq={next_seq})")

    await session.send_json({
        "type": "resume_ok",
        "utterance_id": uid if isinstance(uid, str) else "",
        "state": state,
        "next_seq": next_seq,
    })
    return True


async def handle_frame(session, device_id, chunk):
    turn = _turns.get(device_id)

    if turn is None or turn.state != "uploading":
        return True                               # frame basi: abaikan

    if len(chunk) <= FRAME_HEADER_BYTES:
        return True

    payload_len = len(chunk) - FRAME_HEADER_BYTES

    if payload_len > WS_MAX_CHUNK_BYTES:
        await send_error(session, "Chunk audio terlalu besar.", "CHUNK_TOO_LARGE")
        return False

    seq = int.from_bytes(chunk[:FRAME_HEADER_BYTES], "little")
    now = time.monotonic()

    if seq < turn.next_seq:
        return True                               # duplikat (kirim ulang)

    if seq > turn.next_seq:                       # ada frame hilang: minta dikirim ulang
        if now - turn.last_nack_at >= NACK_MIN_INTERVAL:
            turn.last_nack_at = now
            print(f"[WS] {device_id}: gap frame (dapat {seq}, butuh {turn.next_seq}), nack")
            await session.send_json({
                "type": "nack",
                "utterance_id": turn.utterance_id,
                "next_seq": turn.next_seq,
            })
        return True

    if len(turn.data) + payload_len > AUDIO_MAX_SIZE:
        await fail_turn(session, turn, "AUDIO_TOO_LARGE", "Ukuran audio melebihi batas.")
        return True

    turn.data.extend(memoryview(chunk)[FRAME_HEADER_BYTES:])
    turn.next_seq += 1
    turn.frames += 1
    turn.updated = now

    gap = now - turn.last_frame_at
    turn.last_frame_at = now
    if gap > turn.max_gap:
        turn.max_gap = gap
    if gap > UPLOAD_GAP_WARN:
        print(f"[WS] {device_id}: jeda antar frame {gap:.2f}s (frame #{turn.next_seq}) - cek WiFi")

    duration = len(turn.data) / (AUDIO_SAMPLE_RATE * turn.bps)
    if duration > AUDIO_MAX_DURATION:
        await fail_turn(session, turn, "AUDIO_TOO_LONG", "Durasi audio melebihi batas.")
        return True

    pending = turn.next_seq - turn.acked_seq
    if pending >= ACK_EVERY_FRAMES or (pending > 0 and now - turn.last_ack_at >= ACK_MAX_INTERVAL):
        turn.acked_seq = turn.next_seq
        turn.last_ack_at = now
        await session.send_json({
            "type": "ack",
            "utterance_id": turn.utterance_id,
            "seq": turn.next_seq,
        })

    return True


async def handle_audio_end(app, session, device_id, data):
    uid = data.get("utterance_id")
    turn = _turns.get(device_id)

    if turn is None or turn.utterance_id != uid:
        await session.send_json({
            "type": "error",
            "code": "INVALID_STATE",
            "message": "Tidak ada ucapan aktif.",
        })
        return True

    if turn.state in ("processing", "done"):      # kirim ulang: konfirmasi saja
        await session.send_json({
            "type": "upload_ok",
            "utterance_id": uid,
            "bytes": data.get("bytes", 0),
        })
        return True

    if turn.state != "uploading":
        await session.send_json({
            "type": "error",
            "code": "INVALID_STATE",
            "message": "Ucapan sudah dibatalkan.",
        })
        return True

    declared_bytes = data.get("bytes")
    declared_frames = data.get("frames")
    declared_crc = data.get("crc32")

    if not all(isinstance(v, int) and v >= 0 for v in (declared_bytes, declared_frames, declared_crc)):
        await send_error(session, "audio_end tidak valid.", "INVALID_END")
        return False

    # Frame belum lengkap: minta kirim ulang dari yang hilang
    if len(turn.data) < declared_bytes or turn.next_seq < declared_frames:
        print(
            f"[WS] {device_id}: audio_end tapi baru {len(turn.data)}/{declared_bytes} byte, nack"
        )
        await session.send_json({
            "type": "nack",
            "utterance_id": uid,
            "next_seq": turn.next_seq,
        })
        return True

    if len(turn.data) != declared_bytes or turn.next_seq != declared_frames:
        await fail_turn(session, turn, "AUDIO_CORRUPT", "Jumlah byte/frame tidak cocok.")
        return True

    if declared_bytes == 0:
        await fail_turn(session, turn, "EMPTY_AUDIO", "Audio kosong.")
        return True

    if declared_bytes % turn.bps != 0:
        await fail_turn(session, turn, "INVALID_PCM", "Ukuran audio tidak valid.")
        return True

    if (zlib.crc32(turn.data) & 0xFFFFFFFF) != declared_crc:
        await fail_turn(session, turn, "AUDIO_CORRUPT", "CRC audio tidak cocok.")
        return True

    elapsed = time.monotonic() - turn.created

    print(
        f"[WS] {device_id}: upload selesai {len(turn.data)} byte, {turn.frames} frame, "
        f"{elapsed:.2f}s ({len(turn.data) / max(elapsed, 0.001) / 1024:.1f} KB/s), "
        f"jeda terpanjang {turn.max_gap:.2f}s"
    )

    turn.state = "processing"
    turn.updated = time.monotonic()

    await session.send_json({
        "type": "upload_ok",
        "utterance_id": uid,
        "bytes": declared_bytes,
    })

    turn.task = asyncio.create_task(run_turn(app, turn))
    return True


async def handle_cancel(session, device_id):
    turn = _turns.pop(device_id, None)

    if turn is not None:
        if turn.task is not None and not turn.task.done():
            turn.task.cancel()
        turn.state = "failed"
        turn.data = bytearray()

    await session.send_json({"type": "cancelled"})
    return True


# ---------------------------------------------------------------- Handler

async def handle_audio_stream(websocket: WebSocket, device_id: str):
    """Manage authenticated audio WebSocket connection."""

    app = websocket.app

    await websocket.accept()

    if not DEVICE_ID_PATTERN.fullmatch(device_id):
        await websocket.close(code=1008)
        return

    if not HIVI_WS_TOKEN:
        await websocket.close(code=1011)
        return

    try:
        await asyncio.wait_for(
            app.state.ws_connection_slots.acquire(),
            timeout=2,
        )
    except asyncio.TimeoutError:
        await websocket.close(code=1013)
        return

    session = None

    try:
        authenticated = await authenticate(websocket)

        if not authenticated:
            return

        # Perangkat yang sama tersambung lagi (mis. setelah reboot) -> koneksi
        # lama kemungkinan sudah mati tanpa FIN. Usir yang lama, jangan tolak yang baru.
        session = DeviceSession(websocket)
        old = _sessions.get(device_id)

        if old is not None:
            print(f"[WS] {device_id}: koneksi lama diusir oleh koneksi baru")
            old.evict.set()

            try:
                await asyncio.wait_for(old.done.wait(), timeout=EVICT_WAIT)
            except asyncio.TimeoutError:
                print(f"[WS] {device_id}: koneksi lama belum selesai, lanjut")

        _sessions[device_id] = session

        await session.send_json({
            "type": "connected",
            "device_id": device_id,
            "session_id": session.session_id,
            "sample_rate": AUDIO_SAMPLE_RATE,
            "format": "pcm_s16le",
            "protocol": PROTOCOL_VERSION,
            "formats": sorted(BYTES_PER_SAMPLE),
            "max_duration": AUDIO_MAX_DURATION,
        })

        print(f"[WS] Device connected: {device_id} session={session.session_id}")

        while True:
            try:
                message = await receive_message(websocket, session)
            except asyncio.TimeoutError:
                # Saat memproses, perangkat memang diam: jangan dianggap idle.
                turn = _turns.get(device_id)
                if turn is not None and turn.state == "processing":
                    continue
                raise

            if message["type"] == "websocket.disconnect":
                break

            if message.get("text") is not None:
                try:
                    data = json.loads(message["text"])
                except json.JSONDecodeError:
                    await send_error(session, "JSON tidak valid.", "INVALID_JSON")
                    break

                if not isinstance(data, dict):
                    await send_error(session, "JSON tidak valid.", "INVALID_JSON")
                    break

                message_type = data.get("type")

                if message_type == "audio_start":
                    keep = await handle_audio_start(session, device_id, data)
                elif message_type == "resume":
                    keep = await handle_resume(session, device_id, data)
                elif message_type == "audio_end":
                    keep = await handle_audio_end(app, session, device_id, data)
                elif message_type == "cancel":
                    keep = await handle_cancel(session, device_id)
                elif message_type == "ping":
                    await session.send_json({"type": "pong"})
                    keep = True
                else:
                    await send_error(session, "Perintah tidak dikenal.", "UNKNOWN_MESSAGE")
                    keep = False

                if not keep:
                    break

            elif message.get("bytes") is not None:
                if not await handle_frame(session, device_id, message["bytes"]):
                    break

    except Evicted:
        print(f"[WS] Connection replaced: {device_id}")

        try:
            await websocket.close(code=1000)
        except Exception:
            pass

    except asyncio.TimeoutError:
        print(f"[WS] Receive idle timeout: {device_id}")

        try:
            await websocket.close(code=1001)
        except Exception:
            pass

    except WebSocketDisconnect:
        print(f"[WS] Device disconnected: {device_id}")

    except Exception:
        print(f"[WS] Error for {device_id}:\n{traceback.format_exc()}")

        try:
            await websocket.send_json({
                "type": "error",
                "code": "INTERNAL_ERROR",
                "message": "Terjadi kesalahan pada server.",
            })
            await websocket.close(code=1011)
        except Exception:
            pass

    finally:
        if session is not None:
            session.dead = True

            if _sessions.get(device_id) is session:
                _sessions.pop(device_id, None)

            session.done.set()

        app.state.ws_connection_slots.release()

        print(f"[WS] Connection closed: {device_id}")