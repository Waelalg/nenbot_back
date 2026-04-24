from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = BASE_DIR / ".env"
try:
    from dotenv import load_dotenv

    load_dotenv(ENV_PATH)
except ModuleNotFoundError:
    # Allows lightweight imports before dependencies are installed.
    pass

DATA_DIR = BASE_DIR / "backend" / "data"
HXH_DIR = DATA_DIR / "hxh"
TEAM_FILE = DATA_DIR / "team" / "team.json"
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(BASE_DIR / "backend" / "chroma_db")))
if not CHROMA_DIR.is_absolute():
    CHROMA_DIR = (BASE_DIR / CHROMA_DIR).resolve()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MEMORY_TURNS = max(5, min(10, int(os.getenv("MEMORY_TURNS", "8"))))
MAX_CONTEXT_CHUNKS = max(4, min(6, int(os.getenv("MAX_CONTEXT_CHUNKS", "5"))))
ALLOWED_SMALLTALK = os.getenv("ALLOWED_SMALLTALK", "true").lower() == "true"
COLLECTION_NAME = "nenbot_hxh"
