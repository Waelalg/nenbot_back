from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator

from ..models.schemas import ChatRequest, ChatResponse
from .chat_service import _is_clearly_unrelated
from .fallback_answer_service import fallback_answer_service
from .intent_service import classify_message, refusal_message
from .llm_service import llm_service
from .memory_service import memory_service
from .prompt_service import build_messages
from .question_type_service import detect_question_type
from .retrieval_service import retrieval_service
from .team_service import team_service

logger = logging.getLogger(__name__)


def _sse(data: dict[str, object], event: str | None = None) -> str:
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _small_chunks(text: str, size: int = 28) -> Iterator[str]:
    for index in range(0, len(text), size):
        yield text[index : index + size]


def _has_followup_pronoun(query: str) -> bool:
    return re.search(r"\b(he|his|him|she|her|they|their|it|that|this)\b", query.lower()) is not None


class StreamingService:
    def stream_chat(self, req: ChatRequest) -> Iterator[str]:
        history = memory_service.get_history(req.session_id)
        classification = classify_message(req.message, history)
        intent = classification.intent
        normalized_query = classification.normalized_query
        detected_entities = classification.detected_entities

        last_hxh_entity = memory_service.get_last_hxh_entity(req.session_id)
        if intent == "hxh_knowledge" and _has_followup_pronoun(normalized_query) and last_hxh_entity:
            if last_hxh_entity not in detected_entities:
                detected_entities = [last_hxh_entity, *detected_entities]
            normalized_query = f"{normalized_query} {last_hxh_entity}".strip()

        question_type = detect_question_type(normalized_query, intent)
        retrieval = None

        if intent == "out_of_scope" and not _is_clearly_unrelated(normalized_query):
            try:
                retrieval = retrieval_service.build_context(normalized_query, detected_entities, question_type)
                if retrieval.has_relevant_context:
                    intent = "hxh_knowledge"
                    question_type = detect_question_type(normalized_query, intent)
            except RuntimeError:
                retrieval = None

        metadata: dict[str, object] = {
            "intent": intent,
            "question_type": question_type,
            "normalized_query": normalized_query,
            "detected_entities": detected_entities,
            "matched_member": None,
            "matched_alias": None,
            "memory_used": bool(history),
            "sources": [],
        }

        try:
            if intent == "out_of_scope":
                answer = refusal_message()
                yield from self._stream_static(req, answer, metadata)
                return

            if intent == "team_info":
                result = team_service.answer_team_question(req.message, memory_service.get_last_team_member(req.session_id))
                answer = result.answer
                if result.matched_member:
                    memory_service.set_last_team_member(req.session_id, result.matched_member)
                metadata.update(
                    {
                        "question_type": detect_question_type(normalized_query, intent, result.matched_member),
                        "matched_member": result.matched_member,
                        "matched_alias": result.matched_alias,
                        "sources": ["team.json"],
                    }
                )
                yield from self._stream_static(req, answer, metadata, matched_member=result.matched_member)
                return

            if intent == "allowed_smalltalk":
                answer = self._smalltalk_answer(req.message, history)
                yield from self._stream_static(req, answer, metadata)
                return

            retrieval = retrieval or retrieval_service.build_context(normalized_query, detected_entities, question_type)
            metadata["sources"] = retrieval.sources
            if not retrieval.context:
                answer = (
                    "I do not have enough Hunter x Hunter information in the local knowledge base to answer that. "
                    "Try re-running ingestion or adding a relevant source file."
                )
                yield from self._stream_static(req, answer, metadata)
                return

            messages = build_messages(
                user_question=normalized_query,
                history=history,
                intent=intent,
                question_type=question_type,
                detected_entities=detected_entities,
                retrieved_context=retrieval.context,
                team_context="",
            )

            full_answer: list[str] = []
            streamed_any_token = False
            yield _sse({"token": ""})
            try:
                for token in llm_service.stream(messages):
                    full_answer.append(token)
                    if token:
                        streamed_any_token = True
                    yield _sse({"token": token})
            except RuntimeError:
                logger.exception("Groq streaming failed; using local Hunter x Hunter fallback answer")
                fallback_answer = fallback_answer_service.build_answer(
                    query=normalized_query,
                    question_type=question_type,
                    retrieved_context=retrieval.context,
                    detected_entities=detected_entities,
                )
                if streamed_any_token and "".join(full_answer).strip():
                    continuation = (
                        "\n\nThe live model response was interrupted, so I completed the answer from the local "
                        "Hunter x Hunter knowledge base.\n\n"
                        + fallback_answer
                    )
                    for chunk in _small_chunks(continuation):
                        yield _sse({"token": chunk})
                    answer = "".join(full_answer).strip() + continuation
                    if detected_entities:
                        memory_service.set_last_hxh_entity(req.session_id, detected_entities[0])
                    memory_service.add_interaction(req.session_id, req.message, answer, intent, detected_entities)
                    yield _sse(metadata, event="metadata")
                    yield _sse({}, event="done")
                    return
                yield from self._stream_static(req, fallback_answer, metadata)
                return

            answer = "".join(full_answer).strip()
            if detected_entities:
                memory_service.set_last_hxh_entity(req.session_id, detected_entities[0])
            memory_service.add_interaction(req.session_id, req.message, answer, intent, detected_entities)
            yield _sse(metadata, event="metadata")
            yield _sse({}, event="done")
        except Exception:
            logger.exception("Streaming chat failed")
            error_answer = "Sorry, NENBOT had a technical issue while generating the answer. Please try again."
            yield _sse({"token": error_answer})
            metadata["intent"] = intent
            yield _sse(metadata, event="metadata")
            yield _sse({}, event="done")

    def _stream_static(
        self,
        req: ChatRequest,
        answer: str,
        metadata: dict[str, object],
        matched_member: str | None = None,
    ) -> Iterator[str]:
        yield _sse({"token": ""})
        for chunk in _small_chunks(answer):
            yield _sse({"token": chunk})
        detected = metadata.get("detected_entities")
        memory_service.add_interaction(
            req.session_id,
            req.message,
            answer,
            str(metadata["intent"]),
            detected if isinstance(detected, list) else [],
            matched_member,
        )
        yield _sse(metadata, event="metadata")
        yield _sse({}, event="done")

    def _smalltalk_answer(self, message: str, history: list[dict[str, str]]) -> str:
        normalized = message.lower()
        if "remember" in normalized:
            recent = [item for item in history[-8:] if item["role"] == "user"]
            if not recent:
                return "I do not have previous user questions in this session yet."
            return "I remember these recent user questions: " + "; ".join(item["content"] for item in recent) + "."
        return (
            "I am NENBOT, a Hunter x Hunter-only assistant. Ask about HxH characters, Nen, arcs, factions, "
            "abilities, lore, or the project team."
        )


streaming_service = StreamingService()

