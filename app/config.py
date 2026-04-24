from pathlib import Path
import os

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent

if (PACKAGE_ROOT / "data").exists():
    BACKEND_DIR = PACKAGE_ROOT
else:
    BACKEND_DIR = REPO_ROOT / "backend"

if (REPO_ROOT / "frontend").exists() and BACKEND_DIR.name == "backend":
    BASE_DIR = REPO_ROOT
else:
    BASE_DIR = BACKEND_DIR

ENV_CANDIDATES = [
    BACKEND_DIR / ".env",
    BASE_DIR / ".env",
]

try:
    from dotenv import load_dotenv

    for env_path in ENV_CANDIDATES:
        if env_path.exists():
            load_dotenv(env_path)
            break
except ModuleNotFoundError:
    # Allows lightweight imports before dependencies are installed.
    pass

DATA_DIR = BACKEND_DIR / "data"
HXH_DIR = DATA_DIR / "hxh"
TEAM_FILE = DATA_DIR / "team" / "team.json"
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(BACKEND_DIR / "chroma_db")))
if not CHROMA_DIR.is_absolute():
    CHROMA_DIR = (BASE_DIR / CHROMA_DIR).resolve()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
VOICE_STT_MODEL = os.getenv("VOICE_STT_MODEL", "whisper-large-v3-turbo")
VOICE_TTS_MODEL = os.getenv("VOICE_TTS_MODEL", "canopylabs/orpheus-v1-english")
VOICE_TTS_VOICE = os.getenv("VOICE_TTS_VOICE", "troy")
VOICE_TTS_FORMAT = os.getenv("VOICE_TTS_FORMAT", "wav").lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MEMORY_TURNS = max(5, min(10, int(os.getenv("MEMORY_TURNS", "8"))))
MAX_CONTEXT_CHUNKS = max(4, min(6, int(os.getenv("MAX_CONTEXT_CHUNKS", "5"))))
ALLOWED_SMALLTALK = os.getenv("ALLOWED_SMALLTALK", "true").lower() == "true"
COLLECTION_NAME = "nenbot_hxh"
