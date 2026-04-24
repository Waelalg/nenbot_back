from __future__ import annotations

import re

from ..models.schemas import ChatResponse
from .intent_service import classify_message, refusal_message
from .llm_service import llm_service
from .memory_service import memory_service
from .question_service import detect_question_type
from .retrieval_service import retrieval_service
from .team_service import team_service
from .prompt_service import build_messages


CLEAR_OUT_OF_SCOPE_TERMS = {
    "naruto", "one piece", "dragon ball", "capital of france", "world cup", "python",
    "javascript", "integral", "derivative", "equation", "football", "soccer",
    "solve", "capital", "france", "zidane",
}


def _is_clearly_unrelated(query: str) -> bool:
    return any(term in query for term in CLEAR_OUT_OF_SCOPE_TERMS)


def _has_followup_pronoun(query: str) -> bool:
    return re.search(r"\b(he|his|him|she|her|they|their|it|that|this)\b", query.lower()) is not None


class ChatService:
    def answer(self, session_id: str, message: str) -> ChatResponse:
        history = memory_service.get_history(session_id)
        classification = classify_message(message, history)
        intent = classification.intent
        normalized_query = classification.normalized_query
        detected_entities = classification.detected_entities
        last_hxh_entity = memory_service.get_last_hxh_entity(session_id)
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

        if intent == "out_of_scope":
            answer = refusal_message()
            memory_service.add_interaction(session_id, message, answer, intent, detected_entities)
            return ChatResponse(
                answer=answer,
                intent=intent,
                question_type=question_type,
                normalized_query=normalized_query,
                detected_entities=detected_entities,
                sources=[],
                memory_used=bool(history),
                session_id=session_id,
            )

        if intent == "team_info":
            result = team_service.answer_team_question(message, memory_service.get_last_team_member(session_id))
            answer = result.answer
            if result.matched_member:
                memory_service.set_last_team_member(session_id, result.matched_member)
            question_type = detect_question_type(normalized_query, intent, result.matched_member)
            memory_service.add_interaction(session_id, message, answer, intent, detected_entities, result.matched_member)
            return ChatResponse(
                answer=answer,
                intent=intent,
                question_type=question_type,
                normalized_query=normalized_query,
                detected_entities=detected_entities,
                matched_member=result.matched_member,
                matched_alias=result.matched_alias,
                sources=["team.json"],
                memory_used=bool(history),
                session_id=session_id,
            )

        if intent == "allowed_smalltalk":
            answer = self._smalltalk_answer(message, history)
            memory_service.add_interaction(session_id, message, answer, intent, detected_entities)
            return ChatResponse(
                answer=answer,
                intent=intent,
                question_type=question_type,
                normalized_query=normalized_query,
                detected_entities=detected_entities,
                sources=[],
                memory_used=bool(history),
                session_id=session_id,
            )

        retrieval = retrieval or retrieval_service.build_context(normalized_query, detected_entities, question_type)
        if not retrieval.context:
            answer = (
                "I do not have enough Hunter x Hunter information in the local knowledge base to answer that. "
                "Try re-running ingestion or adding a relevant source file."
            )
        else:
            messages = build_messages(
                user_question=normalized_query,
                history=history,
                intent=intent,
                question_type=question_type,
                detected_entities=detected_entities,
                retrieved_context=retrieval.context,
                team_context="",
            )
            answer = llm_service.generate(messages)

        if detected_entities:
            memory_service.set_last_hxh_entity(session_id, detected_entities[0])
        memory_service.add_interaction(session_id, message, answer, intent, detected_entities)
        return ChatResponse(
            answer=answer,
            intent=intent,
            question_type=question_type,
            normalized_query=normalized_query,
            detected_entities=detected_entities,
            sources=retrieval.sources,
            memory_used=bool(history),
            session_id=session_id,
        )

    def _smalltalk_answer(self, message: str, history: list[dict[str, str]]) -> str:
        normalized = message.lower()
        if "remember" in normalized:
            if not history:
                return "I do not have previous messages in this session yet."
            recent = [item for item in history[-8:] if item["role"] == "user"]
            if not recent:
                return "I do not have previous user questions in this session yet."
            return "I remember these recent user questions: " + "; ".join(item["content"] for item in recent) + "."
        return (
            "I am NENBOT, a Hunter x Hunter-only assistant. Ask me about characters, Nen, arcs, factions, "
            "abilities, lore, or the project team information."
        )


chat_service = ChatService()


