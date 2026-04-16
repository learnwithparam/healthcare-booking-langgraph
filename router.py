from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import List
import json
import logging

from models import ChatRequest, Doctor, Appointment, BookingResponse, ServiceInfo
from service import (
    chat_turn,
    get_doctors,
    get_patient_appointments,
    reset_conversation,
    new_thread_id,
)
from utils.llm_provider import get_provider_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/healthcare-booking", tags=["healthcare-booking"])


@router.get("/health", response_model=ServiceInfo)
async def health_check():
    return ServiceInfo(
        status="healthy",
        service="healthcare-booking-langgraph",
        description=(
            "Conversational booking agent built as a LangGraph state machine "
            "with SQLite-backed appointments."
        ),
    )


@router.get("/provider-info")
async def provider_info():
    try:
        config = get_provider_config()
        return {"provider_name": config["provider_name"], "model": config["model"]}
    except Exception as e:
        return {"provider_name": "unknown", "model": "unknown", "error": str(e)}


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Run one conversational turn through the LangGraph state machine and
    stream every node transition as Server-Sent Events.
    """
    thread_id = request.thread_id or new_thread_id()

    async def event_stream():
        try:
            yield f"data: {json.dumps({'type': 'start', 'thread_id': thread_id})}\n\n"
            async for event in chat_turn(
                thread_id=thread_id,
                user_message=request.message,
                patient_id=request.patient_id,
            ):
                yield f"data: {json.dumps(event)}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            logger.exception("chat_stream error")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/doctors", response_model=List[Doctor])
async def doctors_endpoint():
    return [Doctor(**d) for d in get_doctors()]


@router.get("/appointments/{patient_id}", response_model=List[Appointment])
async def appointments_endpoint(patient_id: str):
    rows = get_patient_appointments(patient_id)
    return [Appointment(**r) for r in rows]


@router.post("/reset", response_model=BookingResponse)
async def reset_endpoint(request: ChatRequest):
    if not request.thread_id:
        raise HTTPException(status_code=400, detail="thread_id is required to reset")
    reset_conversation(request.thread_id)
    return BookingResponse(
        thread_id=request.thread_id,
        status="reset",
        message="Conversation cleared. Start a new booking when you're ready.",
    )
