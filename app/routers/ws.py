from fastapi import APIRouter, WebSocket

from app.services.audio_stream import handle_audio_stream

router = APIRouter(tags=["websocket"])


@router.websocket("/ws/audio/{device_id}")
async def websocket_audio(websocket: WebSocket, device_id: str):
    await handle_audio_stream(websocket, device_id)