"""
Service layer between the FastAPI router and the LangGraph booking graph.

Keeps the router thin and lets the graph stay focused on orchestration.
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional, List, Dict, Any, AsyncGenerator

from booking_graph import run_turn, reset_thread
from db_utils import (
    init_db,
    list_all_doctors,
    list_appointments_for_patient,
)

logger = logging.getLogger(__name__)

# Ensure DB is ready at import time
init_db()


def new_thread_id() -> str:
    return f"thr_{uuid.uuid4().hex[:12]}"


async def chat_turn(
    thread_id: Optional[str],
    user_message: str,
    patient_id: Optional[str] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    tid = thread_id or new_thread_id()
    async for event in run_turn(tid, user_message, patient_id=patient_id):
        yield event


def get_doctors() -> List[Dict[str, Any]]:
    return list_all_doctors()


def get_patient_appointments(patient_id: str) -> List[Dict[str, Any]]:
    return list_appointments_for_patient(patient_id)


def reset_conversation(thread_id: str) -> None:
    reset_thread(thread_id)
