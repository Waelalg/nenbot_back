from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .routes.chat import router as chat_router


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

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_BUILD_CANDIDATES = [
    PACKAGE_ROOT.parent / "frontend" / "dist",
    PACKAGE_ROOT / "frontend" / "dist",
    PACKAGE_ROOT / "dist",
]

FRONTEND_BUILD_DIR = next((path for path in FRONTEND_BUILD_CANDIDATES if path.exists()), FRONTEND_BUILD_CANDIDATES[0])
FRONTEND_INDEX = FRONTEND_BUILD_DIR / "index.html"
FRONTEND_CONFIG = FRONTEND_BUILD_DIR / "config.js"
FRONTEND_ASSETS = FRONTEND_BUILD_DIR / "assets"

if FRONTEND_ASSETS.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_ASSETS)), name="assets")


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


