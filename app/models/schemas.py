from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


Intent = Literal["hxh_knowledge", "team_info", "allowed_smalltalk", "out_of_scope"]
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


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=120)
    message: str = Field(..., min_length=1, max_length=2000)

    @field_validator("session_id", "message")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Value cannot be empty.")
        return cleaned


class ChatResponse(BaseModel):
    answer: str
    intent: Intent
    question_type: QuestionType
    normalized_query: str
    detected_entities: list[str] = Field(default_factory=list)
    matched_member: str | None = None
    matched_alias: str | None = None
    sources: list[str] = Field(default_factory=list)
    memory_used: bool
    session_id: str


class ResetRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=120)

    @field_validator("session_id")
    @classmethod
    def strip_session_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Session id cannot be empty.")
        return cleaned


class ResetResponse(BaseModel):
    status: str
    session_id: str


class HealthResponse(BaseModel):
    status: str
    service: str
    model: str
    embedding_model: str
    memory_turns: int
    indexed_chunks: int | None = None


class TeamMember(BaseModel):
    full_name: str
    academic_level: str = "Not provided"
    age: int | str = "Not provided"
    university_name: str = "Not provided"
    field_of_study: str = "Not provided"
    aliases: list[str] = Field(default_factory=list)
