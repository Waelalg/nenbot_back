from collections import defaultdict
from backend.app.config import MEMORY_TURNS

# Stores full messages, trimmed to the last MEMORY_TURNS user-assistant pairs.
_memory_store: dict[str, list[dict[str, str]]] = defaultdict(list)


def get_history(session_id: str) -> list[dict[str, str]]:
    return _memory_store.get(session_id, [])


def add_turn(session_id: str, user_message: str, assistant_message: str) -> None:
    history = _memory_store[session_id]
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": assistant_message})
    max_messages = MEMORY_TURNS * 2
    _memory_store[session_id] = history[-max_messages:]


def reset_session(session_id: str) -> None:
    _memory_store.pop(session_id, None)
