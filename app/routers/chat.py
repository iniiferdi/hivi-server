from uuid import uuid4

from fastapi import APIRouter, HTTPException

from app.schemas import ChatRequest
from app.services.brain import generate_response
from app.services.conversation import get_history, save_exchange

router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post("/chat")
async def chat(payload: ChatRequest):
    session_id = payload.session_id or str(uuid4())

    try:
        history = await get_history(session_id)

        answer = await generate_response(
            message=payload.message,
            history=history,
        )

        await save_exchange(
            session_id=session_id,
            user_message=payload.message,
            assistant_message=answer,
        )

        return {
            "service": "hivi-brain",
            "session_id": session_id,
            "message": payload.message,
            "response": answer,
        }

    except Exception as error:
        print(f"AI Brain error: {error}")
        raise HTTPException(
            status_code=502,
            detail="AI Brain sedang mengalami gangguan.",
        )