from __future__ import annotations

import logging

from openai import OpenAI

from ..config import GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL

logger = logging.getLogger(__name__)


class LLMService:
    def __init__(self) -> None:
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        if not GROQ_API_KEY:
            raise RuntimeError("Missing GROQ_API_KEY. Add it to .env before using generated answers.")
        if self._client is None:
            self._client = OpenAI(
                api_key=GROQ_API_KEY,
                base_url=GROQ_BASE_URL,
                max_retries=3,
                timeout=45.0,
            )
        return self._client

    def generate(self, messages: list[dict[str, str]]) -> str:
        try:
            response = self._get_client().chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.15,
                max_tokens=900,
            )
        except Exception as exc:
            logger.exception("Groq generation failed")
            raise RuntimeError("The Groq API request failed. Check the API key, model name, and network connection.") from exc

        answer = (response.choices[0].message.content or "").strip()
        if not answer:
            raise RuntimeError("The model returned an empty response.")
        return answer

    def stream(self, messages: list[dict[str, str]]):
        try:
            stream = self._get_client().chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.15,
                max_tokens=900,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as exc:
            logger.exception("Groq streaming failed")
            raise RuntimeError("The Groq streaming request failed. Check the API key, model name, and network connection.") from exc


llm_service = LLMService()

