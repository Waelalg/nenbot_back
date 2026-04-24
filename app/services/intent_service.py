from __future__ import annotations

from dataclasses import dataclass

from ..config import ALLOWED_SMALLTALK
from ..data.hxh_aliases import HXH_KEYWORDS, basic_normalize, contains_any, normalize_hxh_query
from ..models.schemas import Intent
from .team_service import team_service


@dataclass(frozen=True)
class ClassificationResult:
    intent: Intent
    normalized_query: str
    detected_entities: list[str]


SMALLTALK_PATTERNS = {
    "hello", "hi", "hey", "salam", "help", "what can you do", "what can you answer",
    "what can i ask", "how do i use", "usage", "who are you", "what are you",
    "what do you remember", "clear memory",
}

SHORT_SMALLTALK = {"hello", "hi", "hey", "salam", "help"}

FOLLOWUP_MARKERS = {
    "he", "she", "they", "it", "his", "her", "their", "that", "those", "this",
    "them", "him", "which", "what about",
}


def _history_has_hxh(history: list[dict[str, str]]) -> bool:
    joined = " ".join(item.get("content", "") for item in history[-8:]).lower()
    return contains_any(joined, HXH_KEYWORDS)


def classify_message(message: str, history: list[dict[str, str]] | None = None) -> ClassificationResult:
    raw_normalized = basic_normalize(message)
    normalized, detected_entities = normalize_hxh_query(message)
    history = history or []

    if ALLOWED_SMALLTALK and raw_normalized in SHORT_SMALLTALK:
        return ClassificationResult("allowed_smalltalk", normalized, detected_entities)

    if team_service.detect_team_intent(message):
        return ClassificationResult("team_info", normalized, detected_entities)

    if detected_entities or contains_any(normalized, HXH_KEYWORDS):
        return ClassificationResult("hxh_knowledge", normalized, detected_entities)

    first_words = " ".join(normalized.split()[:2])
    is_short_followup = len(normalized.split()) <= 14 and (
        (normalized.split()[:1] and normalized.split()[0] in FOLLOWUP_MARKERS)
        or first_words in FOLLOWUP_MARKERS
    )
    if is_short_followup and _history_has_hxh(history):
        return ClassificationResult("hxh_knowledge", normalized, detected_entities)

    if ALLOWED_SMALLTALK and contains_any(normalized, SMALLTALK_PATTERNS):
        return ClassificationResult("allowed_smalltalk", normalized, detected_entities)

    return ClassificationResult("out_of_scope", normalized, detected_entities)


def classify_intent(message: str, history: list[dict[str, str]] | None = None) -> Intent:
    return classify_message(message, history).intent


def refusal_message() -> str:
    return (
        "I'm specialized only in Hunter x Hunter and project team information. "
        "I can help with Hunter x Hunter lore, characters, Nen, arcs, factions, "
        "abilities, and the team details stored for this project."
    )


