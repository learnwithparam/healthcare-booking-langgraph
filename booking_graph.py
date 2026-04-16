"""
Booking graph
=============

A pure-LLM conversational state machine built with LangGraph. The graph
shapes a multi-turn booking flow into six small nodes that share a single
BookingState TypedDict:

    intent_router
         |
         v
    collect_symptoms  ->  match_specialty  ->  find_doctor
                                                    |
                                                    v
                                              propose_slots
                                                    |
                                                    v
                                            confirm_booking

Every decision (intent classification, specialty mapping, confirmation
parsing) is delegated to the LLM. There is no regex, no hardcoded keyword
lookup. SQLite is consulted for doctor data and appointment persistence
through helpers in db_utils.
"""

from __future__ import annotations

import json
import logging
from typing import TypedDict, Optional, List, Dict, Any, AsyncGenerator

from langgraph.graph import StateGraph, END

from utils.llm_provider import get_llm_provider
from db_utils import (
    list_doctors_by_specialty,
    list_slots,
    create_appointment,
    get_doctor,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class BookingState(TypedDict, total=False):
    """Shared state passed between every node in the graph."""
    # Conversation
    user_message: str
    thread_id: str
    patient_id: Optional[str]
    history: List[Dict[str, str]]  # [{role, content}, ...]

    # Derived fields
    intent: Optional[str]          # book | list | cancel | smalltalk | unknown
    symptom: Optional[str]         # short free-text description
    specialty: Optional[str]       # e.g. cardiology, dermatology
    doctor: Optional[Dict[str, Any]]
    slot_options: List[str]
    slot: Optional[str]            # chosen slot string
    status: str                    # gathering | ready_to_confirm | confirmed | ended

    # Output
    assistant_reply: str
    events: List[Dict[str, Any]]   # node-level trace for SSE streaming


SUPPORTED_SPECIALTIES = [
    "general_medicine",
    "cardiology",
    "dermatology",
    "pediatrics",
    "orthopedics",
]


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _record_event(state: BookingState, node: str, detail: Dict[str, Any]) -> None:
    events = state.get("events") or []
    events.append({"node": node, **detail})
    state["events"] = events


async def _llm_json(prompt: str, system: str) -> Dict[str, Any]:
    """
    Call the LLM and parse a JSON object out of the response.

    We keep this deliberately tolerant: the LLM occasionally wraps JSON in
    fences, so we strip them before parsing.
    """
    provider = get_llm_provider()
    full_prompt = f"[SYSTEM]\n{system}\n\n[USER]\n{prompt}"
    raw = await provider.generate_text(full_prompt, temperature=0.2, max_tokens=400)
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to recover the first {...} block
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
        logger.warning("LLM JSON parse failed, raw=%r", raw[:200])
        return {}


async def _llm_text(prompt: str, system: str) -> str:
    provider = get_llm_provider()
    full_prompt = f"[SYSTEM]\n{system}\n\n[USER]\n{prompt}"
    raw = await provider.generate_text(full_prompt, temperature=0.5, max_tokens=300)
    return (raw or "").strip()


# ---------------------------------------------------------------------------
# Node: intent_router
# ---------------------------------------------------------------------------

async def intent_router(state: BookingState) -> BookingState:
    system = (
        "You classify patient messages for a healthcare booking assistant. "
        "Respond with strict JSON: {\"intent\": one of "
        "[book, list, cancel, smalltalk, unknown]}."
    )
    prompt = (
        "Message: "
        + state.get("user_message", "")
        + "\nReturn JSON only."
    )
    data = await _llm_json(prompt, system)
    intent = data.get("intent") or "unknown"
    state["intent"] = intent
    _record_event(state, "intent_router", {"intent": intent})
    return state


def route_from_intent(state: BookingState) -> str:
    intent = state.get("intent")
    if intent == "book":
        # Continue gathering symptoms unless we already have them
        if state.get("symptom"):
            return "match_specialty"
        return "collect_symptoms"
    # For list/cancel/smalltalk/unknown we short-circuit to a friendly reply.
    return "smalltalk"


# ---------------------------------------------------------------------------
# Node: collect_symptoms
# ---------------------------------------------------------------------------

async def collect_symptoms(state: BookingState) -> BookingState:
    system = (
        "You extract the chief complaint from a patient message. "
        "Return JSON: {\"symptom\": short phrase or null, "
        "\"needs_followup\": bool, \"followup_question\": string}."
    )
    prompt = (
        "Patient said: "
        + state.get("user_message", "")
        + "\nIf the symptom is clear, set needs_followup=false. "
        "Otherwise craft a single short followup question."
    )
    data = await _llm_json(prompt, system)
    symptom = data.get("symptom")
    if symptom:
        state["symptom"] = symptom
    if data.get("needs_followup"):
        state["assistant_reply"] = data.get("followup_question") or (
            "Could you tell me a bit more about what's bothering you?"
        )
        state["status"] = "gathering"
    _record_event(state, "collect_symptoms", {"symptom": state.get("symptom")})
    return state


# ---------------------------------------------------------------------------
# Node: match_specialty
# ---------------------------------------------------------------------------

async def match_specialty(state: BookingState) -> BookingState:
    system = (
        "You map a chief complaint to one medical specialty from this list: "
        + ", ".join(SUPPORTED_SPECIALTIES)
        + ". Return JSON: {\"specialty\": one of the list}."
    )
    prompt = (
        "Complaint: "
        + (state.get("symptom") or state.get("user_message", ""))
        + "\nPick the single best specialty."
    )
    data = await _llm_json(prompt, system)
    specialty = data.get("specialty")
    if specialty not in SUPPORTED_SPECIALTIES:
        specialty = "general_medicine"
    state["specialty"] = specialty
    _record_event(state, "match_specialty", {"specialty": specialty})
    return state


# ---------------------------------------------------------------------------
# Node: find_doctor
# ---------------------------------------------------------------------------

async def find_doctor(state: BookingState) -> BookingState:
    specialty = state.get("specialty") or "general_medicine"
    doctors = list_doctors_by_specialty(specialty)
    if not doctors:
        doctors = list_doctors_by_specialty("general_medicine")
    # Deterministic pick: most experienced doctor first.
    state["doctor"] = doctors[0] if doctors else None
    _record_event(
        state,
        "find_doctor",
        {"doctor": state["doctor"]["name"] if state.get("doctor") else None},
    )
    return state


# ---------------------------------------------------------------------------
# Node: propose_slots
# ---------------------------------------------------------------------------

async def propose_slots(state: BookingState) -> BookingState:
    doctor = state.get("doctor")
    if not doctor:
        state["assistant_reply"] = (
            "I couldn't find a matching doctor right now. Would you like to try another specialty?"
        )
        state["status"] = "ended"
        return state

    slots = list_slots(doctor["id"], count=3)
    state["slot_options"] = slots

    bullet = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(slots))
    state["assistant_reply"] = (
        f"I'd recommend {doctor['name']} ({doctor['specialty'].replace('_', ' ')}, "
        f"{doctor['years_experience']} years experience).\n"
        f"Here are the next available slots:\n{bullet}\n"
        "Reply with the slot number to confirm."
    )
    state["status"] = "ready_to_confirm"
    _record_event(state, "propose_slots", {"slots": slots})
    return state


# ---------------------------------------------------------------------------
# Node: confirm_booking
# ---------------------------------------------------------------------------

async def confirm_booking(state: BookingState) -> BookingState:
    options = state.get("slot_options") or []
    if not options:
        state["assistant_reply"] = "Let's pick a doctor first. What symptom are you experiencing?"
        state["status"] = "gathering"
        return state

    system = (
        "You interpret which slot a patient chose from a numbered list. "
        "Return JSON: {\"choice_index\": 1-based int or null, \"declined\": bool}."
    )
    prompt = (
        "Slots:\n"
        + "\n".join(f"{i+1}. {s}" for i, s in enumerate(options))
        + f"\nPatient reply: {state.get('user_message', '')}"
    )
    data = await _llm_json(prompt, system)
    if data.get("declined"):
        state["assistant_reply"] = "No problem. Tell me when you'd like to try again."
        state["status"] = "ended"
        return state

    idx = data.get("choice_index")
    if not isinstance(idx, int) or idx < 1 or idx > len(options):
        state["assistant_reply"] = "Which slot works for you? Reply with 1, 2, or 3."
        state["status"] = "ready_to_confirm"
        return state

    chosen = options[idx - 1]
    doctor = state.get("doctor") or {}
    patient_id = state.get("patient_id") or state.get("thread_id") or "anonymous"
    appt = create_appointment(
        patient_id=patient_id,
        doctor_id=doctor.get("id", 0),
        doctor_name=doctor.get("name", "Unknown"),
        specialty=doctor.get("specialty", "general_medicine"),
        slot=chosen,
    )
    state["slot"] = chosen
    state["status"] = "confirmed"
    state["assistant_reply"] = (
        f"Booked! Appointment #{appt['id']} with {appt['doctor_name']} on {chosen}. "
        "You'll get a reminder by email."
    )
    _record_event(state, "confirm_booking", {"appointment_id": appt["id"]})
    return state


# ---------------------------------------------------------------------------
# Fallback: smalltalk / list / unknown
# ---------------------------------------------------------------------------

async def smalltalk(state: BookingState) -> BookingState:
    intent = state.get("intent") or "unknown"
    system = (
        "You are a friendly healthcare booking assistant. Keep replies to 1-2 sentences."
    )
    prompt = (
        f"The patient's intent was classified as '{intent}'. "
        f"Message: {state.get('user_message', '')}\n"
        "Reply helpfully. If they want to book, invite them to describe their symptom."
    )
    state["assistant_reply"] = await _llm_text(prompt, system)
    state["status"] = state.get("status") or "gathering"
    _record_event(state, "smalltalk", {})
    return state


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(BookingState)

    graph.add_node("intent_router", intent_router)
    graph.add_node("collect_symptoms", collect_symptoms)
    graph.add_node("match_specialty", match_specialty)
    graph.add_node("find_doctor", find_doctor)
    graph.add_node("propose_slots", propose_slots)
    graph.add_node("confirm_booking", confirm_booking)
    graph.add_node("smalltalk", smalltalk)

    graph.set_entry_point("intent_router")

    graph.add_conditional_edges(
        "intent_router",
        route_from_intent,
        {
            "collect_symptoms": "collect_symptoms",
            "match_specialty": "match_specialty",
            "smalltalk": "smalltalk",
        },
    )

    def after_collect(state: BookingState) -> str:
        if state.get("status") == "gathering" and not state.get("symptom"):
            return "end"
        return "match_specialty"

    graph.add_conditional_edges(
        "collect_symptoms",
        after_collect,
        {"match_specialty": "match_specialty", "end": END},
    )

    graph.add_edge("match_specialty", "find_doctor")
    graph.add_edge("find_doctor", "propose_slots")
    graph.add_edge("propose_slots", END)
    graph.add_edge("confirm_booking", END)
    graph.add_edge("smalltalk", END)

    return graph.compile()


_COMPILED = None


def get_graph():
    global _COMPILED
    if _COMPILED is None:
        _COMPILED = build_graph()
    return _COMPILED


# ---------------------------------------------------------------------------
# Thread memory
# ---------------------------------------------------------------------------

# Simple in-process memory. For production use a LangGraph checkpointer.
_THREADS: Dict[str, BookingState] = {}


def get_thread_state(thread_id: str) -> BookingState:
    state = _THREADS.get(thread_id)
    if state is None:
        state = BookingState(
            thread_id=thread_id,
            history=[],
            slot_options=[],
            events=[],
            status="gathering",
            assistant_reply="",
        )
        _THREADS[thread_id] = state
    return state


def reset_thread(thread_id: str) -> None:
    _THREADS.pop(thread_id, None)


async def run_turn(
    thread_id: str,
    user_message: str,
    patient_id: Optional[str] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Run one conversational turn through the graph and yield SSE-ready events.

    Routing:
      - If the previous turn left the thread in 'ready_to_confirm', we send
        this turn to confirm_booking directly.
      - Otherwise we enter at intent_router.
    """
    state = get_thread_state(thread_id)
    state["user_message"] = user_message
    state["patient_id"] = patient_id
    state["events"] = []
    state["assistant_reply"] = ""

    graph = get_graph()

    if state.get("status") == "ready_to_confirm":
        result = await confirm_booking(state)
    else:
        result = await graph.ainvoke(state)

    # Persist state back
    _THREADS[thread_id] = dict(result)  # type: ignore[arg-type]

    # Stream each node event, then final reply
    for ev in result.get("events", []) or []:
        yield {"type": "node", **ev}

    yield {
        "type": "final",
        "thread_id": thread_id,
        "status": result.get("status"),
        "reply": result.get("assistant_reply", ""),
    }
