"""
main.py
=======
FastAPI backend for the Socratic AI Teaching Assistant platform.

Architecture:
  - Communicates with a locally-running vLLM instance via the OpenAI-compatible
    HTTP API (http://localhost:8000/v1).
  - Exposes a Server-Sent Events (SSE) endpoint at POST /chat/stream for
    real-time streaming responses to the Next.js frontend.
  - Stateless per-request: conversation history is maintained on the client
    side and sent with each request.

Run:
    uvicorn backend.main:app --host 0.0.0.0 --port 8080 --workers 1
"""

import asyncio
import logging
import os
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI, APIConnectionError, APIStatusError
from pydantic import BaseModel, Field, field_validator

from .prompts import build_assistant_message, build_system_message, build_user_message

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("socratic_backend")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VLLM_BASE_URL: str = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY: str = os.getenv("VLLM_API_KEY", "EMPTY")  # vLLM accepts any non-empty key
MODEL_NAME: str = os.getenv("MODEL_NAME", "socratic_awq_q4")

ALLOWED_ORIGINS: list[str] = os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:3000"
).split(",")

# Generation parameters
MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "1024"))
TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
TOP_P: float = float(os.getenv("TOP_P", "0.9"))
REPETITION_PENALTY: float = float(os.getenv("REPETITION_PENALTY", "1.1"))

# ---------------------------------------------------------------------------
# OpenAI Client (pointing at local vLLM)
# ---------------------------------------------------------------------------

openai_client = AsyncOpenAI(
    base_url=VLLM_BASE_URL,
    api_key=VLLM_API_KEY,
)

# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Socratic AI Teaching Assistant API",
    description=(
        "Backend for the Advanced Algorithms course Socratic tutor. "
        "Streams responses from a locally-served AWQ-quantised Llama-3-8B model via vLLM."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------


class ConversationMessage(BaseModel):
    """A single message in the conversation history."""

    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    """
    Request body for the /chat/stream endpoint.

    Attributes:
        messages: Full conversation history (excluding system prompt).
                  The system prompt is prepended server-side to prevent
                  prompt injection attacks from the client.
        session_id: Optional client-side session identifier for logging.
    """

    messages: list[ConversationMessage] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Conversation history (user and assistant turns only)",
    )
    session_id: str | None = Field(
        default=None,
        max_length=64,
        description="Optional session ID for server-side logging",
    )

    @field_validator("messages")
    @classmethod
    def last_message_must_be_user(
        cls, v: list[ConversationMessage]
    ) -> list[ConversationMessage]:
        """Ensure the conversation ends with a user message."""
        if v and v[-1].role != "user":
            raise ValueError("The last message in the conversation must be from the user.")
        return v


# ---------------------------------------------------------------------------
# SSE Streaming Generator
# ---------------------------------------------------------------------------


async def stream_chat_response(
    messages: list[dict[str, str]],
    session_id: str | None,
) -> AsyncGenerator[str, None]:
    """
    Streams token chunks from the vLLM OpenAI-compatible API as SSE events.

    Each chunk is formatted as:
        data: <token_text>\n\n

    The stream terminates with:
        data: [DONE]\n\n

    Args:
        messages: Full message list including system prompt.
        session_id: Optional session ID for logging.

    Yields:
        SSE-formatted string chunks.

    Raises:
        Propagates APIConnectionError and APIStatusError as HTTP 503/502.
    """
    logger.info(
        "Starting stream | session=%s | messages=%d",
        session_id or "anonymous",
        len(messages),
    )

    try:
        stream = await openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            extra_body={"repetition_penalty": REPETITION_PENALTY},
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                # SSE format: data: <content>\n\n
                # Escape newlines within the token to prevent premature SSE boundary
                escaped = delta.content.replace("\n", "\\n")
                yield f"data: {escaped}\n\n"

        yield "data: [DONE]\n\n"
        logger.info("Stream complete | session=%s", session_id or "anonymous")

    except APIConnectionError as exc:
        logger.error("vLLM connection error: %s", exc)
        yield "data: [ERROR] Could not connect to the model server.\n\n"
        yield "data: [DONE]\n\n"

    except APIStatusError as exc:
        logger.error("vLLM API error: status=%d body=%s", exc.status_code, exc.body)
        yield f"data: [ERROR] Model server returned status {exc.status_code}.\n\n"
        yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post(
    "/chat/stream",
    summary="Stream a Socratic tutor response",
    response_description="Server-Sent Events stream of token chunks",
)
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """
    Accepts a conversation history and returns a streaming SSE response from
    the Socratic AI tutor.

    The system prompt is injected server-side to prevent client-side tampering.
    The vLLM backend handles continuous batching across all concurrent sessions.

    Args:
        request: ChatRequest containing conversation history and optional session_id.

    Returns:
        StreamingResponse with media_type='text/event-stream'.
    """
    # Build full message list: system prompt + conversation history
    full_messages: list[dict[str, str]] = [build_system_message()]
    for msg in request.messages:
        if msg.role == "user":
            full_messages.append(build_user_message(msg.content))
        else:
            full_messages.append(build_assistant_message(msg.content))

    return StreamingResponse(
        stream_chat_response(full_messages, request.session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable Nginx buffering for SSE
            "Connection": "keep-alive",
        },
    )


@app.get("/health", summary="Health check")
async def health_check() -> dict[str, str]:
    """
    Simple liveness check. Does not validate vLLM connectivity.

    Returns:
        JSON with status 'ok'.
    """
    return {"status": "ok", "model": MODEL_NAME}


@app.get("/readiness", summary="Readiness check — verifies vLLM connectivity")
async def readiness_check() -> dict[str, str]:
    """
    Attempts to list models from the vLLM OpenAI-compatible API to verify
    that the model server is reachable and ready to serve requests.

    Returns:
        JSON with status 'ready' and available model list, or raises HTTP 503.
    """
    try:
        models = await openai_client.models.list()
        model_ids = [m.id for m in models.data]
        return {"status": "ready", "available_models": ", ".join(model_ids)}
    except APIConnectionError as exc:
        logger.warning("Readiness check failed: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"vLLM server at {VLLM_BASE_URL} is not reachable: {exc}",
        ) from exc
