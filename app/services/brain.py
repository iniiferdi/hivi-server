import asyncio
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

REQUEST_TIMEOUT_MS = 30_000
MAX_HISTORY_ITEMS = 10

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY belum dikonfigurasi.")


def _env_flag(name: str):
    """True/False bila variabel diset, None bila tidak diset."""

    raw = os.getenv(name)

    if raw is None:
        return None

    return raw.strip().lower() in {"1", "true", "yes", "on"}


# Kunci berawalan "AQ." adalah API key Vertex AI Express Mode, BUKAN kunci
# Gemini Developer API (AIza...). Keduanya memakai backend berbeda.
# Paksa lewat GEMINI_USE_VERTEX=1 atau GEMINI_USE_VERTEX=0 bila perlu.
_vertex_flag = _env_flag("GEMINI_USE_VERTEX")
USE_VERTEX = _vertex_flag if _vertex_flag is not None else API_KEY.startswith("AQ.")

_http_options = types.HttpOptions(timeout=REQUEST_TIMEOUT_MS)

if USE_VERTEX:
    client = genai.Client(
        vertexai=True,
        api_key=API_KEY,
        http_options=_http_options,
    )
else:
    client = genai.Client(
        api_key=API_KEY,
        http_options=_http_options,
    )

print(
    f"[BRAIN] Client siap: model={MODEL} "
    f"backend={'vertex-express' if USE_VERTEX else 'gemini-developer'}"
)

HIVI_INSTRUCTIONS = """
Kamu adalah Hivi, asisten virtual pribadi berbahasa Indonesia.

Karakteristik:
- Berbicara secara natural, ramah, dan sopan.
- Menggunakan bahasa Indonesia sebagai bahasa utama.
- Memberikan jawaban yang jelas dan tidak bertele-tele.
- Memahami konteks percakapan sebelumnya.
- Tidak mengarang informasi ketika tidak mengetahui jawabannya.
- Menggunakan gaya percakapan yang nyaman untuk interaksi suara.

Jawablah seperti asisten virtual yang sedang berbicara
langsung dengan penggunanya.
"""

GENERATION_CONFIG = types.GenerateContentConfig(
    system_instruction=HIVI_INSTRUCTIONS,
    temperature=0.7,
)

ROLE_MAP = {
    "user": "user",
    "assistant": "model",
}


def build_contents(
    message: str,
    history: list[dict] | None,
) -> list[types.Content]:
    """Convert chat history and the new message into Gemini contents."""

    contents: list[types.Content] = []

    for item in (history or [])[-MAX_HISTORY_ITEMS:]:
        role = ROLE_MAP.get(item.get("role"))
        text = item.get("content")

        if role is None or not isinstance(text, str) or not text.strip():
            continue

        contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=text)],
            )
        )

    contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=message)],
        )
    )

    return contents


def _empty_reason(response) -> str:
    """Alasan respons kosong (diblokir, dipotong, dll.) untuk pesan error."""

    parts = []

    feedback = getattr(response, "prompt_feedback", None)
    if feedback is not None:
        parts.append(f"prompt_feedback={feedback}")

    for candidate in getattr(response, "candidates", None) or []:
        parts.append(f"finish_reason={getattr(candidate, 'finish_reason', None)}")

    return "; ".join(parts) or "tanpa detail"


async def generate_response(
    message: str,
    history: list[dict] | None = None,
) -> str:
    """Generate Hivi's reply for the given user message.

    Error dilempar apa adanya; pemanggil (audio_stream) mencatat traceback
    lengkap dan memutuskan retry/fallback.
    """

    response = await client.aio.models.generate_content(
        model=MODEL,
        contents=build_contents(message, history),
        config=GENERATION_CONFIG,
    )

    result = (response.text or "").strip()

    if not result:
        raise RuntimeError(
            f"Gemini menghasilkan respons kosong ({_empty_reason(response)})."
        )

    return result


async def check_brain(timeout: float = 20.0) -> str:
    """Tes singkat saat startup agar kunci/model yang salah langsung ketahuan."""

    return await asyncio.wait_for(
        generate_response("Jawab dengan satu kata saja: siap"),
        timeout=timeout,
    )