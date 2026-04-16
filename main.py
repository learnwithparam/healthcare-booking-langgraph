from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from router import router

app = FastAPI(
    title="Conversational State Machines with LangGraph",
    description="Build a healthcare booking agent as a pure-LLM conversational state machine with LangGraph. Route intents, match patients to doctors, suggest appointment slots, and persist bookings to SQLite - all through one streaming FastAPI endpoint.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
async def root():
    return {"service": "healthcare-booking-langgraph", "docs": "/docs"}
