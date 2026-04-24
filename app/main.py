from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from app.routes.chat import router as chat_router


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

app = FastAPI(
    title="NENBOT API",
    version="1.0.0",
    description="Hunter x Hunter-only RAG chatbot API with short-term memory and structured team lookup.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = PROJECT_ROOT / "frontend"
FRONTEND_INDEX = FRONTEND_DIR / "index.html"
FRONTEND_CONFIG = FRONTEND_DIR / "config.js"

if FRONTEND_DIR.exists() and FRONTEND_INDEX.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/", include_in_schema=False)
def root():
    if FRONTEND_INDEX.exists():
        return FileResponse(FRONTEND_INDEX)
    return {
        "service": "NENBOT API",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/config.js", include_in_schema=False)
def frontend_config():
    if FRONTEND_CONFIG.exists():
        return FileResponse(FRONTEND_CONFIG, media_type="application/javascript")
    return Response('window.NENBOT_API_BASE = window.NENBOT_API_BASE || "";', media_type="application/javascript")
