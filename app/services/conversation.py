import os

import json
import redis.asyncio as redis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

MAX_MESSAGES = 20
SESSION_TTL = 7 * 24 * 60 * 60  # 7 hari

redis_client: redis.Redis | None = None


async def init_redis():
    global redis_client

    redis_client = redis.from_url(
        REDIS_URL,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5,
    )

    await redis_client.ping()
    print("Hivi Redis Connection is ready!")


async def close_redis():
    global redis_client

    if redis_client is not None:
        await redis_client.aclose()
        redis_client = None


def get_session_key(session_id: str) -> str:
    return f"hivi:conversation:{session_id}"


async def get_history(session_id: str) -> list[dict]:
    if redis_client is None:
        raise RuntimeError("Redis belum diinisialisasi.")

    key = get_session_key(session_id)

    messages = await redis_client.lrange(
        key,
        -MAX_MESSAGES,
        -1,
    )

    if messages:
        await redis_client.expire(key, SESSION_TTL)

    return [
        json.loads(message)
        for message in messages
    ]


async def save_exchange(
    session_id: str,
    user_message: str,
    assistant_message: str,
):
    if redis_client is None:
        raise RuntimeError("Redis belum diinisialisasi.")

    key = get_session_key(session_id)

    messages = [
        json.dumps({
            "role": "user",
            "content": user_message,
        }),
        json.dumps({
            "role": "assistant",
            "content": assistant_message,
        }),
    ]

    async with redis_client.pipeline(transaction=True) as pipe:
        pipe.rpush(key, *messages)
        pipe.ltrim(key, -MAX_MESSAGES, -1)
        pipe.expire(key, SESSION_TTL)
        await pipe.execute()


async def clear_session(session_id: str):
    if redis_client is None:
        raise RuntimeError("Redis belum diinisialisasi.")

    await redis_client.delete(
        get_session_key(session_id)
    )
