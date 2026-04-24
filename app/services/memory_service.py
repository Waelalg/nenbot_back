from __future__ import annotations

from collections import defaultdict
from copy import deepcopy

from backend.app.config import MEMORY_TURNS


class MemoryService:
    """In-memory session store for the last N user-assistant turns."""

    def __init__(self, max_turns: int = MEMORY_TURNS) -> None:
        self.max_turns = max(1, max_turns)
        self._store: dict[str, list[dict[str, str]]] = defaultdict(list)
        self._state: dict[str, dict[str, str]] = defaultdict(dict)

    def get_history(self, session_id: str) -> list[dict[str, str]]:
        return deepcopy(self._store.get(session_id, []))

    def add_interaction(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        intent: str | None = None,
        detected_entities: list[str] | None = None,
        matched_member: str | None = None,
    ) -> None:
        history = self._store[session_id]
        history.append(
            {
                "role": "user",
                "content": user_message,
                "intent": intent or "",
                "detected_entities": ", ".join(detected_entities or []),
                "matched_member": matched_member or "",
            }
        )
        history.append({"role": "assistant", "content": assistant_message, "intent": intent or ""})
        self._store[session_id] = history[-self.max_turns * 2 :]

    def add_turn(self, session_id: str, user_message: str, assistant_message: str) -> None:
        self.add_interaction(session_id, user_message, assistant_message)

    def reset(self, session_id: str) -> None:
        self._store.pop(session_id, None)
        self._state.pop(session_id, None)

    def get_state(self, session_id: str, key: str) -> str | None:
        return self._state.get(session_id, {}).get(key)

    def set_state(self, session_id: str, key: str, value: str) -> None:
        self._state[session_id][key] = value

    def get_last_hxh_entity(self, session_id: str) -> str | None:
        return self.get_state(session_id, "last_hxh_entity")

    def set_last_hxh_entity(self, session_id: str, value: str) -> None:
        self.set_state(session_id, "last_hxh_entity", value)

    def get_last_team_member(self, session_id: str) -> str | None:
        return self.get_state(session_id, "last_team_member")

    def set_last_team_member(self, session_id: str, value: str) -> None:
        self.set_state(session_id, "last_team_member", value)

    def clear_session(self, session_id: str) -> None:
        self.reset(session_id)


memory_service = MemoryService()
