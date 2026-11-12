import asyncio
import os
import re

from dotenv import load_dotenv
from ollama import AsyncClient

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

REQUEST_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "60"))   # detik
KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")           # model tetap di memori
NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
MAX_NEW_TOKENS = int(os.getenv("OLLAMA_MAX_TOKENS", "300"))  # jawaban suara harus singkat
MAX_HISTORY_ITEMS = 10

client = AsyncClient(host=OLLAMA_HOST, timeout=REQUEST_TIMEOUT)

print(f"[BRAIN] Client siap: model={MODEL} backend=ollama host={OLLAMA_HOST}")

HIVI_INSTRUCTIONS = """
Kamu adalah Hivi, teman belajar yang ramah dan sabar. Kamu bukan asisten
atau pelayan, melainkan teman yang belajar bersama penggunanya.

Karakter:
- Berbicara santai, hangat, dan akrab, dengan bahasa Indonesia sehari-hari yang sopan.
- Antusias dan menyemangati. Beri apresiasi saat pengguna berhasil atau mau mencoba.
- Sabar. Kalau pengguna salah atau bingung, jangan menyalahkan; bantu pelan-pelan.

Cara membantu belajar:
- Jelaskan konsep dengan bahasa sederhana, pakai contoh atau perumpamaan dari kehidupan sehari-hari.
- Untuk soal atau masalah, bimbing langkah demi langkah dan beri petunjuk dulu,
  jangan langsung memberi jawaban akhir kecuali pengguna memintanya.
- Sesekali ajukan satu pertanyaan singkat untuk mengecek pemahaman,
  tetapi jangan di setiap jawaban.
- Kalau pengguna salah paham, luruskan dengan lembut.
- Jika tidak tahu atau tidak yakin, katakan jujur dan jangan mengarang jawaban.
- Ingat konteks percakapan sebelumnya dan hubungkan dengan hal yang sudah dipelajari.

Aturan format (jawabanmu akan dibacakan oleh text-to-speech):
- Jawab singkat, maksimal 3 sampai 4 kalimat. Pecah penjelasan panjang menjadi beberapa giliran.
- Jangan gunakan markdown, tanda bintang, tanda pagar, daftar bernomor, atau emoji.
- Tulis angka, simbol, dan rumus dengan kata-kata yang mudah diucapkan
  (misalnya "dua tambah tiga sama dengan lima").

Berbicaralah seperti teman yang sedang mengobrol langsung saat belajar bersama.
""".strip()

ALLOWED_ROLES = {"user", "assistant"}


def build_messages(
    message: str,
    history: list[dict] | None,
) -> list[dict]:
    """Convert chat history and the new message into Ollama chat messages."""

    messages = [{"role": "system", "content": HIVI_INSTRUCTIONS}]

    for item in (history or [])[-MAX_HISTORY_ITEMS:]:
        role = item.get("role")
        text = item.get("content")

        if role not in ALLOWED_ROLES or not isinstance(text, str) or not text.strip():
            continue

        messages.append({"role": role, "content": text})

    messages.append({"role": "user", "content": message})

    return messages


_MARKDOWN_CHARS = re.compile(r"[*_#`>~]+")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")


def clean_for_speech(text: str) -> str:
    """Buang sisa markdown agar tidak terbaca aneh oleh TTS."""

    text = _MARKDOWN_CHARS.sub("", text)
    text = _MULTI_SPACE.sub(" ", text)

    return text.strip()


async def generate_response(
    message: str,
    history: list[dict] | None = None,
) -> str:
    """Generate Hivi's reply for the given user message.

    Error dilempar apa adanya; pemanggil (audio_stream) mencatat traceback
    lengkap dan memutuskan retry/fallback.
    """

    response = await client.chat(
        model=MODEL,
        messages=build_messages(message, history),
        options={
            "temperature": 0.7,
            "num_ctx": NUM_CTX,
            "num_predict": MAX_NEW_TOKENS,
        },
        keep_alive=KEEP_ALIVE,
    )

    result = clean_for_speech(response["message"]["content"] or "")

    if not result:
        raise RuntimeError(
            f"Ollama menghasilkan respons kosong "
            f"(done_reason={response.get('done_reason')})."
        )

    return result


async def check_brain(timeout: float = 20.0) -> str:
    """Tes singkat + pemanasan model (memuat model ke memori)."""

    return await asyncio.wait_for(
        generate_response("Jawab dengan satu kata saja: siap"),
        timeout=timeout,
    )