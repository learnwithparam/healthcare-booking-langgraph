from pydantic import BaseModel, Field
from typing import Optional, List


class ChatRequest(BaseModel):
    """Incoming chat turn from a patient."""
    message: str = Field(..., description="User utterance for this turn")
    thread_id: Optional[str] = Field(
        default=None,
        description="Conversation thread id. Omit to start a new conversation.",
    )
    patient_id: Optional[str] = Field(
        default=None,
        description="Stable patient identifier (used when persisting an appointment).",
    )


class Doctor(BaseModel):
    """A doctor available for booking."""
    id: int
    name: str
    specialty: str
    years_experience: int


class Appointment(BaseModel):
    """A persisted appointment row."""
    id: int
    patient_id: str
    doctor_id: int
    doctor_name: str
    specialty: str
    slot: str
    status: str
    created_at: str


class BookingResponse(BaseModel):
    """Non-streaming booking response (used by /reset etc)."""
    thread_id: str
    status: str
    message: str


class ServiceInfo(BaseModel):
    """Health / identity response."""
    status: str
    service: str
    description: str
