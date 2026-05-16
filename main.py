"""
main.py
---------------------------------------------------------
FastAPI app for SHL Assessment Recommender
---------------------------------------------------------

Endpoints:
GET  /health
POST /chat

Features:
✅ FastAPI
✅ FAISS semantic retrieval
✅ Gemini integration
✅ Strict response schema
✅ Error handling
✅ CORS enabled
"""

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, field_validator

import retriever
import agent


# ---------------------------------------------------------
# LOGGING
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# FASTAPI STARTUP
# ---------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info(
        "Initializing retriever..."
    )

    retriever.initialize(
        "catalog.json"
    )

    logger.info(
        "Retriever initialized."
    )

    yield

    logger.info(
        "Application shutdown."
    )


# ---------------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------------

app = FastAPI(
    title="SHL Assessment Recommender",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# REQUEST / RESPONSE MODELS
# ---------------------------------------------------------

class Message(BaseModel):

    role: str
    content: str

    @field_validator("role")
    @classmethod
    def validate_role(cls, v):

        if v not in (
            "user",
            "assistant"
        ):

            raise ValueError(
                "role must be "
                "'user' or 'assistant'"
            )

        return v


class ChatRequest(BaseModel):

    messages: list[Message]

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v):

        if not v:

            raise ValueError(
                "messages cannot be empty"
            )

        return v


class Recommendation(BaseModel):

    name: str
    url: str


class ChatResponse(BaseModel):

    reply: str

    recommendations: list[
        Recommendation
    ]

    end_of_conversation: bool


# ---------------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "ok"
    }


# ---------------------------------------------------------
# CHAT ENDPOINT
# ---------------------------------------------------------

@app.post(
    "/chat",
    response_model=ChatResponse
)
def chat(
    request: ChatRequest
):

    # Convert Pydantic models
    messages = [
        {
            "role": m.role,
            "content": m.content,
        }
        for m in request.messages
    ]

    # Assignment limit
    if len(messages) > 8:

        raise HTTPException(
            status_code=400,
            detail=(
                "Conversation exceeds "
                "maximum 8 turns."
            ),
        )

    try:

        result = agent.chat(
            messages
        )

    except Exception as e:

        logger.error(
            f"Agent error: {e}",
            exc_info=True
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Internal agent error."
            ),
        )

    # Safety defaults
    recommendations = (
        result.get(
            "recommendations"
        ) or []
    )

    return ChatResponse(
        reply=result.get(
            "reply",
            ""
        ),

        recommendations=[
            Recommendation(
                name=r.get(
                    "name",
                    ""
                ),
                url=r.get(
                    "url",
                    ""
                ),
            )
            for r in recommendations
        ],

        end_of_conversation=bool(
            result.get(
                "end_of_conversation",
                False
            )
        ),
    )


# ---------------------------------------------------------
# LOCAL DEV SERVER
# ---------------------------------------------------------

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )