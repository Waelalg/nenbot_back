from __future__ import annotations

import re
from typing import Literal

from app.models.schemas import Intent


QuestionType = Literal[
    "identity",
    "definition",
    "simple_explanation",
    "detailed_explanation",
    "list",
    "comparison",
    "arc_summary",
    "ability",
    "relationship",
    "team_profile",
    "team_field",
    "memory_followup",
    "allowed_smalltalk",
    "out_of_scope",
]


def _has_pronoun_followup(text: str) -> bool:
    return bool(re.search(r"\b(he|his|him|she|her|they|their|it|that|this member)\b", text))


def detect_question_type(query: str, intent: Intent, matched_member: str | None = None) -> QuestionType:
    text = query.lower().strip()

    if intent == "out_of_scope":
        return "out_of_scope"
    if intent == "allowed_smalltalk":
        return "allowed_smalltalk"
    if _has_pronoun_followup(text):
        return "memory_followup"

    if intent == "team_info":
        field_terms = ["age", "old", "university", "school", "academic", "level", "field", "study", "studies"]
        if matched_member and any(term in text for term in field_terms):
            return "team_field"
        return "team_profile"

    if any(term in text for term in ["compare", "difference between", "versus", " vs "]):
        return "comparison"
    if any(text.startswith(prefix) for prefix in ["list", "name all", "what are the"]) or "types" in text or "categories" in text:
        return "list"
    if any(term in text for term in ["arc", "summarize", "what happened"]):
        return "arc_summary"
    if any(term in text for term in ["relationship", "friend", "father", "mother", "hate", "why does", "who is ging to"]):
        return "relationship"
    if any(term in text for term in ["ability", "abilities", "bungee gum", "godspeed", "chain jail", "emperor time", "jajanken", "how does"]):
        return "ability"
    if any(term in text for term in ["explain simply", "simple", "simply", "for beginner", "beginner"]):
        return "simple_explanation"
    if any(term in text for term in ["in detail", "detailed", "deeply", "explain"]) and "simply" not in text:
        return "detailed_explanation"
    if text.startswith(("what is", "what are", "define", "describe")):
        return "definition"
    if text.startswith(("who is", "who are", "tell me about", "do you know")):
        return "identity"
    return "definition"
