from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from backend.app.config import EMBEDDING_MODEL, GROQ_MODEL, MEMORY_TURNS
from backend.app.models.schemas import ChatRequest, ChatResponse, HealthResponse, ResetRequest, ResetResponse
from backend.app.rag.ingest import ingest_summary
from backend.app.rag.vector_store import count_documents
from backend.app.services.chat_service import chat_service
from backend.app.services.memory_service import memory_service
from backend.app.services.streaming_service import streaming_service
from backend.app.services.team_service import team_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        indexed_chunks = count_documents()
    except Exception:
        indexed_chunks = None
    return HealthResponse(
        status="ok",
        service="nenbot-api",
        model=GROQ_MODEL,
        embedding_model=EMBEDDING_MODEL,
        memory_turns=MEMORY_TURNS,
        indexed_chunks=indexed_chunks,
    )


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        return chat_service.answer(req.session_id, req.message)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected chat failure")
        raise HTTPException(status_code=500, detail="Unexpected chat error.") from exc


@router.post("/chat/stream")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        streaming_service.stream_chat(req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest) -> ResetResponse:
    memory_service.reset(req.session_id)
    return ResetResponse(status="ok", session_id=req.session_id)


@router.post("/ingest")
def reingest() -> dict[str, int | str]:
    try:
        summary = ingest_summary()
    except Exception as exc:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc
    return {"status": "ok", **summary}


@router.get("/team")
def team() -> dict[str, object]:
    team_service.clear_cache()
    return {"members": [member.model_dump() for member in team_service.members()]}


@router.get("/sources")
def sources() -> dict[str, object]:
    return {"indexed_chunks": count_documents()}
