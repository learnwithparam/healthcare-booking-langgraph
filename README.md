# Conversational State Machines with LangGraph

![learnwithparam.com](https://www.learnwithparam.com/ai-bootcamp/opengraph-image)

Build a healthcare booking agent as a pure-LLM conversational state machine with LangGraph. Route intents, match patients to doctors, suggest appointment slots, and persist bookings to SQLite, all through one streaming FastAPI endpoint.

> Start learning at [learnwithparam.com](https://learnwithparam.com). Regional pricing available with discounts of up to 60%.

## What You'll Learn

- Model a multi-turn conversation as a LangGraph state machine with typed shared state
- Route patient intents with a pure-LLM classifier (no regex, no keyword matching)
- Share a single `BookingState` TypedDict across every node in the graph
- Query a SQLite-backed catalog of doctors and slots from inside graph nodes
- Stream a multi-step booking flow to the client over Server-Sent Events
- Persist confirmed appointments and retrieve them by patient id

## Tech Stack

- **FastAPI** - High-performance async Python web framework
- **LangGraph** - State machine orchestration for the conversation flow
- **LangChain** - LLM integration primitives
- **SQLite** - Embedded appointments and doctors database
- **Pydantic** - Request and response validation
- **LLM Provider Pattern** - Supports Fireworks, OpenRouter, Gemini, OpenAI
- **Docker** - Containerized development

## Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (installed automatically by `make setup`)
- An API key from any supported LLM provider

### Quick Start

```bash
# One command to set up and run
make dev

# Or step by step:
make setup          # Create .env and install dependencies
# Edit .env with your API key
make run            # Start the FastAPI server
```

### With Docker

```bash
make build          # Build the Docker image
make up             # Start the container
make logs           # View logs
make down           # Stop the container
```

### API Documentation

Once running, open [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive Swagger UI.

Primary endpoints:

- `POST /healthcare-booking/chat/stream` - send a patient turn, receive SSE events
- `GET /healthcare-booking/doctors` - list the seeded doctor catalog
- `GET /healthcare-booking/appointments/{patient_id}` - list a patient's appointments
- `POST /healthcare-booking/reset` - clear a conversation thread

## Challenges

Work through these incrementally to build the full application:

1. **Model the shared state** - Define `BookingState` as a TypedDict with `intent`, `patient`, `symptom`, `doctor`, `slot`, and `status`
2. **Build the intent router node** - Classify every turn into `book`, `list`, `cancel`, `smalltalk`, or `unknown` using the LLM
3. **Map symptoms to specialties** - Add a node that picks one specialty from the supported list for any chief complaint
4. **Query doctors and slots from SQLite** - Bootstrap the database, seed doctors, and surface available slots from inside a graph node
5. **Confirm and persist the booking** - Add a `confirm_booking` node that interprets the patient's slot choice and writes an appointment row
6. **Stream the flow over SSE** - Expose `POST /healthcare-booking/chat/stream` with `thread_id` memory so each turn resumes the graph from its last state

## Makefile Targets

```
make help           Show all available commands
make setup          Initial setup (create .env, install deps)
make dev            Setup and run (one command!)
make run            Start FastAPI server
make build          Build Docker image
make up             Start container
make down           Stop container
make clean          Remove venv and cache
```

## Learn more

- Start the course: [learnwithparam.com/courses/conversational-state-machines-langgraph](https://www.learnwithparam.com/courses/conversational-state-machines-langgraph)
- AI Bootcamp for Software Engineers: [learnwithparam.com/ai-bootcamp](https://www.learnwithparam.com/ai-bootcamp)
- All courses: [learnwithparam.com/courses](https://www.learnwithparam.com/courses)
