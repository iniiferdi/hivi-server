import asyncio
import os
import time
from pathlib import Path

import aiosqlite

DB_PATH = os.getenv("SQLITE_PATH", "data/hivi.db")

MAX_MESSAGES = 20
SESSION_TTL = 7 * 24 * 60 * 60  # 7 hari
CLEANUP_INTERVAL = 60 * 60      # bersihkan sesi kedaluwarsa maks. 1x per jam

db: aiosqlite.Connection | None = None
db_lock = asyncio.Lock()
_last_cleanup = 0.0


def _require_db() -> aiosqlite.Connection:
    if db is None:
        raise RuntimeError("Database belum diinisialisasi.")
    return db


async def init_db():
    global db, _last_cleanup

    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

    db = await aiosqlite.connect(DB_PATH)

    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA synchronous=NORMAL")
    await db.execute("PRAGMA foreign_keys=ON")
    await db.execute("PRAGMA busy_timeout=5000")

    await db.executescript(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            updated_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL
                REFERENCES sessions(session_id) ON DELETE CASCADE,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at REAL NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_messages_session
            ON messages(session_id, id);

        CREATE INDEX IF NOT EXISTS idx_sessions_updated
            ON sessions(updated_at);
        """
    )
    await db.commit()

    await _cleanup_expired(force=True)
    _last_cleanup = time.time()

    print("Hivi SQLite Connection is ready!")


async def close_db():
    global db

    if db is not None:
        await db.close()
        db = None


async def _cleanup_expired(force: bool = False):
    """Hapus sesi yang tidak aktif lebih dari SESSION_TTL (pesan ikut terhapus via CASCADE)."""
    global _last_cleanup

    now = time.time()
    if not force and now - _last_cleanup < CLEANUP_INTERVAL:
        return

    conn = _require_db()
    await conn.execute(
        "DELETE FROM sessions WHERE updated_at < ?",
        (now - SESSION_TTL,),
    )
    await conn.commit()
    _last_cleanup = now


async def get_history(session_id: str) -> list[dict]:
    conn = _require_db()

    async with db_lock:
        cursor = await conn.execute(
            """
            SELECT role, content FROM (
                SELECT id, role, content
                FROM messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
            )
            ORDER BY id ASC
            """,
            (session_id, MAX_MESSAGES),
        )
        rows = await cursor.fetchall()
        await cursor.close()

        if rows:
            # Perpanjang TTL, setara dengan EXPIRE di Redis
            await conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (time.time(), session_id),
            )
            await conn.commit()

    return [{"role": role, "content": content} for role, content in rows]


async def _save_exchange(
    session_id: str,
    user_message: str,
    assistant_message: str,
):
    conn = _require_db()
    now = time.time()

    async with db_lock:
        try:
            await conn.execute(
                """
                INSERT INTO sessions (session_id, updated_at)
                VALUES (?, ?)
                ON CONFLICT(session_id) DO UPDATE SET updated_at = excluded.updated_at
                """,
                (session_id, now),
            )

            await conn.executemany(
                """
                INSERT INTO messages (session_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (session_id, "user", user_message, now),
                    (session_id, "assistant", assistant_message, now),
                ],
            )

            await conn.execute(
                """
                DELETE FROM messages
                WHERE session_id = ?
                  AND id NOT IN (
                      SELECT id FROM messages
                      WHERE session_id = ?
                      ORDER BY id DESC
                      LIMIT ?
                  )
                """,
                (session_id, session_id, MAX_MESSAGES),
            )

            await conn.commit()
        except Exception:
            await conn.rollback()
            raise

        await _cleanup_expired()


async def save_exchange(
    session_id: str,
    user_message: str,
    assistant_message: str,
):
    # shield: bila task pemanggil dibatalkan (ucapan baru / cancel),
    # penulisan tetap selesai sehingga tidak ada transaksi menggantung.
    await asyncio.shield(
        _save_exchange(session_id, user_message, assistant_message)
    )


async def clear_session(session_id: str):
    conn = _require_db()

    async with db_lock:
        await conn.execute(
            "DELETE FROM sessions WHERE session_id = ?",
            (session_id,),
        )
        await conn.commit()