"""
FastAPI application for hosting a WebSocket-based chat interface.

This module provides a FastAPI application that serves as a WebSocket server for real-time
communication. It includes endpoints for serving the main web interface
and handling WebSocket connections for chat interactions.

Exercise routing:
- Exercise 1 (RAG) → priority
- Exercise 0 (Simple agent with file context)
- Hardcoded fallback (demo safety net)
"""

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.requests import Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from util.logger_config import logger
from util.configuration import settings, PROJECT_ROOT


# ==========================================================
# CONFIG
# ==========================================================

# Toggle for workshop
USE_EXERCISE_1 = True


# ==========================================================
# EXERCISE 1 – RAG AGENT
# ==========================================================

EXERCISE_1_AVAILABLE = False
try:
    from agents.hotel_rag_agent import answer_hotel_question_rag
    EXERCISE_1_AVAILABLE = True
    logger.info("✅ Exercise 1 (RAG) agent loaded successfully")
except Exception as e:
    logger.warning(f"Exercise 1 agent not available: {e}")
    EXERCISE_1_AVAILABLE = False


# ==========================================================
# EXERCISE 0 – SIMPLE AGENT
# ==========================================================

EXERCISE_0_AVAILABLE = False
try:
    from agents.hotel_simple_agent import handle_hotel_query_simple, load_hotel_data

    try:
        load_hotel_data()
        EXERCISE_0_AVAILABLE = True
        logger.info("✅ Exercise 0 agent loaded successfully and hotel data verified")
    except Exception as e:
        logger.warning(f"Exercise 0 agent loaded but data not ready: {e}")
        EXERCISE_0_AVAILABLE = False

except Exception as e:
    logger.warning(f"Exercise 0 agent not available: {e}")
    EXERCISE_0_AVAILABLE = False


# ==========================================================
# HARDCODED FALLBACK RESPONSES (DEMO SAFETY NET)
# ==========================================================

HARDCODED_RESPONSES = {
    "list the hotels in france": """Here are the hotels in France:

**Paris:**
- Grand Victoria
- Majestic Plaza
- Obsidian Tower

**Nice:**
- Imperial Crown
- Royal Sovereign""",
}


def find_matching_response(query: str) -> str:
    query_lower = query.lower().strip()

    if query_lower in HARDCODED_RESPONSES:
        return HARDCODED_RESPONSES[query_lower]

    return """I'm a demo API with hardcoded responses.

Try asking:
- "List the hotels in France"
- "Room prices in Paris"
- "Meal plans in Nice"

*This is a workshop fallback response.*"""


# ==========================================================
# FASTAPI LIFESPAN
# ==========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting AI Hospitality API...")
    yield
    logger.info("Shutting down AI Hospitality API...")


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "static")), name="static")
templates = Jinja2Templates(directory=str(PROJECT_ROOT / "templates"))


# ==========================================================
# HTTP ENDPOINT
# ==========================================================

@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ==========================================================
# WEBSOCKET ENDPOINT
# ==========================================================

@app.websocket("/ws/{uuid}")
async def websocket_endpoint(websocket: WebSocket, uuid: str):
    await websocket.accept()
    logger.info("WebSocket connection opened for %s", uuid)

    try:
        while True:
            data = await websocket.receive_text()
            logger.info("Received from %s: %s", uuid, data)

            try:
                payload = json.loads(data)
                user_query = payload.get("content", data)
            except json.JSONDecodeError:
                user_query = data

            # ==================================================
            # ROUTING LOGIC
            # ==================================================

            try:
                if USE_EXERCISE_1 and EXERCISE_1_AVAILABLE:
                    logger.info("➡️ Using Exercise 1 (RAG) agent")
                    response_content = answer_hotel_question_rag(user_query)

                elif EXERCISE_0_AVAILABLE:
                    logger.info("➡️ Using Exercise 0 (Simple) agent")
                    response_content = await handle_hotel_query_simple(user_query)

                else:
                    logger.info("➡️ Using hardcoded fallback")
                    response_content = find_matching_response(user_query)

            except Exception as e:
                logger.error("Agent error: %s", e, exc_info=True)
                response_content = find_matching_response(user_query)

            await websocket.send_text(
                f"JSONSTART{json.dumps({'role': 'assistant', 'content': response_content})}JSONEND"
            )

            logger.info("Sent response to %s", uuid)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for %s", uuid)

    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ==========================================================
# LOCAL RUN
# ==========================================================

if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting server on {settings.API_HOST}:{settings.API_PORT}")
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
    )
