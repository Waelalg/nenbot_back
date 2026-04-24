from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass
from io import BytesIO

from openai import OpenAI
from PIL import Image, ImageOps, UnidentifiedImageError

from ..config import GROQ_API_KEY, GROQ_BASE_URL, GROQ_VISION_MODEL
from ..data.hxh_aliases import normalize_hxh_query
from .fallback_answer_service import fallback_answer_service
from .retrieval_service import retrieval_service

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/gif",
}
MAX_COMPRESSED_IMAGE_BYTES = 2_800_000
MAX_IMAGE_DIMENSION = 1600

VISION_SYSTEM_PROMPT = """You are NENBOT's Hunter x Hunter image recognizer.

You must identify only Hunter x Hunter subjects from images.

Return only valid JSON with this exact schema:
{
  "is_hxh_character": true or false,
  "recognized_entity": string or null,
  "entity_type": "character" | "group" | "unknown",
  "confidence": "high" | "medium" | "low",
  "reason": "short reason",
  "top_guesses": ["guess 1", "guess 2", "guess 3"]
}

Rules:
- Recognize only Hunter x Hunter characters, groups, or unmistakable Hunter x Hunter subjects.
- If the image is not clearly from Hunter x Hunter, set is_hxh_character to false.
- If the image is too unclear, set recognized_entity to null or your best guess, set confidence to low, and provide up to 3 top_guesses.
- Prefer canonical names such as Killua Zoldyck, Gon Freecss, Kurapika, Leorio Paradinight, Hisoka Morow, Chrollo Lucilfer, Isaac Netero, Meruem, Neferpitou, Illumi Zoldyck, Alluka Zoldyck, Nanika, Phantom Troupe, or Zoldyck Family.
- Never identify another anime character as Hunter x Hunter.
"""


@dataclass(frozen=True)
class VisionRecognition:
    is_hxh_character: bool
    recognized_entity: str | None
    entity_type: str
    confidence: str
    reason: str
    top_guesses: list[str]
    follow_up_suggestions: list[str]
    answer: str
    sources: list[str]
    detected_entities: list[str]


class VisionService:
    def __init__(self) -> None:
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        if not GROQ_API_KEY:
            raise RuntimeError("Missing GROQ_API_KEY. Add it to .env before using image recognition.")
        if self._client is None:
            self._client = OpenAI(
                api_key=GROQ_API_KEY,
                base_url=GROQ_BASE_URL,
                max_retries=2,
                timeout=45.0,
            )
        return self._client

    def identify_hxh_image(self, image_bytes: bytes, content_type: str | None = None) -> VisionRecognition:
        normalized_bytes, mime_type = self._prepare_image(image_bytes, content_type)
        payload = self._run_vision_model(normalized_bytes, mime_type)

        is_hxh_character = bool(payload.get("is_hxh_character"))
        raw_entity = (payload.get("recognized_entity") or "").strip() or None
        entity_type = self._sanitize_entity_type(payload.get("entity_type"))
        confidence = self._sanitize_confidence(payload.get("confidence"))
        reason = str(payload.get("reason") or "").strip()
        top_guesses = self._normalize_guesses(payload.get("top_guesses"))

        if raw_entity:
            canonical_entity, detected_entities = self._canonicalize_entity(raw_entity)
            if canonical_entity and canonical_entity not in top_guesses:
                top_guesses = [canonical_entity, *top_guesses][:3]
        else:
            canonical_entity = None
            detected_entities = []

        if not is_hxh_character or not canonical_entity:
            answer = (
                "I cannot confidently identify this image as a Hunter x Hunter character. "
                "NENBOT only recognizes Hunter x Hunter characters and groups from images."
            )
            if reason:
                answer += f" {reason}"
            if top_guesses:
                answer += "\n\nTop Hunter x Hunter guesses:\n" + "\n".join(
                    f"{index + 1}. {guess}" for index, guess in enumerate(top_guesses[:3])
                )
            return VisionRecognition(
                is_hxh_character=False,
                recognized_entity=None,
                entity_type="unknown",
                confidence=confidence,
                reason=reason,
                top_guesses=top_guesses[:3],
                follow_up_suggestions=self._follow_up_suggestions(None, "unknown"),
                answer=answer,
                sources=[],
                detected_entities=[],
            )

        retrieval = retrieval_service.build_context(
            canonical_entity,
            detected_entities,
            "identity",
        )
        answer_prefix = self._answer_prefix(canonical_entity, entity_type, confidence)
        if retrieval.context:
            profile = fallback_answer_service.build_answer(
                query=f"who is {canonical_entity}",
                question_type="identity",
                retrieved_context=retrieval.context,
                detected_entities=detected_entities,
            )
            answer = f"{answer_prefix}\n\n{profile}".strip()
            if confidence == "low" and top_guesses:
                answer += "\n\nTop Hunter x Hunter guesses:\n" + "\n".join(
                    f"{index + 1}. {guess}" for index, guess in enumerate(top_guesses[:3])
                )
            sources = retrieval.sources
        else:
            answer = (
                f"{answer_prefix}\n\nI recognized the Hunter x Hunter subject, but I do not have enough local "
                "knowledge-base context to build a grounded profile yet."
            )
            if top_guesses:
                answer += "\n\nTop Hunter x Hunter guesses:\n" + "\n".join(
                    f"{index + 1}. {guess}" for index, guess in enumerate(top_guesses[:3])
                )
            sources = []

        return VisionRecognition(
            is_hxh_character=True,
            recognized_entity=canonical_entity,
            entity_type=entity_type,
            confidence=confidence,
            reason=reason,
            top_guesses=top_guesses[:3],
            follow_up_suggestions=self._follow_up_suggestions(canonical_entity, entity_type),
            answer=answer,
            sources=sources,
            detected_entities=detected_entities,
        )

    def _prepare_image(self, image_bytes: bytes, content_type: str | None) -> tuple[bytes, str]:
        if not image_bytes:
            raise RuntimeError("The uploaded image was empty.")

        if content_type and content_type.lower() not in SUPPORTED_IMAGE_TYPES:
            raise RuntimeError("Unsupported image format. Upload JPG, PNG, WEBP, or GIF.")

        try:
            image = Image.open(BytesIO(image_bytes))
            image = ImageOps.exif_transpose(image)
        except UnidentifiedImageError as exc:
            raise RuntimeError("The uploaded file is not a valid image.") from exc

        if image.mode not in {"RGB", "L"}:
            image = image.convert("RGB")
        elif image.mode == "L":
            image = image.convert("RGB")

        if max(image.size) > MAX_IMAGE_DIMENSION:
            image.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION))

        for quality in (92, 88, 82, 76, 70, 64, 58):
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=quality, optimize=True)
            normalized = buffer.getvalue()
            if len(normalized) <= MAX_COMPRESSED_IMAGE_BYTES:
                return normalized, "image/jpeg"

        raise RuntimeError(
            "The image is too large to send to the vision model. Use a smaller or more tightly cropped image."
        )

    def _run_vision_model(self, image_bytes: bytes, mime_type: str) -> dict[str, object]:
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        content = [
            {
                "type": "text",
                "text": (
                    "Identify the Hunter x Hunter character or group shown in this image. "
                    "If the image is not clearly from Hunter x Hunter, say so. Return JSON only."
                ),
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{image_b64}",
                },
            },
        ]

        client = self._get_client()
        try:
            response = client.chat.completions.create(
                model=GROQ_VISION_MODEL,
                messages=[
                    {"role": "system", "content": VISION_SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                temperature=0.1,
                max_tokens=250,
                response_format={"type": "json_object"},
            )
        except Exception:
            logger.exception("Groq vision request with JSON mode failed; retrying without response_format")
            try:
                response = client.chat.completions.create(
                    model=GROQ_VISION_MODEL,
                    messages=[
                        {"role": "system", "content": VISION_SYSTEM_PROMPT},
                        {"role": "user", "content": content},
                    ],
                    temperature=0.1,
                    max_tokens=250,
                )
            except Exception as exc:
                logger.exception("Groq vision request failed")
                raise RuntimeError(
                    "NENBOT could not analyze the image right now. Check the Groq vision model setting and API access."
                ) from exc

        raw_content = (response.choices[0].message.content or "").strip()
        if not raw_content:
            raise RuntimeError("The vision model returned an empty response.")
        return self._parse_json_payload(raw_content)

    def _parse_json_payload(self, raw_content: str) -> dict[str, object]:
        try:
            return json.loads(raw_content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw_content, re.S)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
        raise RuntimeError("The vision model response could not be parsed.")

    def _sanitize_confidence(self, value: object) -> str:
        text = str(value or "").strip().lower()
        return text if text in {"high", "medium", "low"} else "unknown"

    def _sanitize_entity_type(self, value: object) -> str:
        text = str(value or "").strip().lower()
        return text if text in {"character", "group", "unknown"} else "unknown"

    def _canonicalize_entity(self, raw_entity: str) -> tuple[str | None, list[str]]:
        _, detected_entities = normalize_hxh_query(raw_entity)
        canonical_entity = detected_entities[0] if detected_entities else raw_entity.strip() or None
        return canonical_entity, [canonical_entity] if canonical_entity else []

    def _normalize_guesses(self, raw_guesses: object) -> list[str]:
        if not isinstance(raw_guesses, list):
            return []

        guesses: list[str] = []
        seen: set[str] = set()
        for raw_guess in raw_guesses[:6]:
            guess_text = str(raw_guess or "").strip()
            if not guess_text:
                continue
            canonical_guess, _ = self._canonicalize_entity(guess_text)
            if not canonical_guess:
                continue
            key = canonical_guess.lower()
            if key in seen:
                continue
            seen.add(key)
            guesses.append(canonical_guess)
            if len(guesses) >= 3:
                break
        return guesses

    def _follow_up_suggestions(self, entity: str | None, entity_type: str) -> list[str]:
        if not entity:
            return []
        if entity_type == "group":
            return [
                f"Who are the main members of {entity}?",
                f"What arc is {entity} important in?",
                f"Why is {entity} important in Hunter x Hunter?",
            ]
        return [
            f"What is {entity}'s Nen type?",
            f"What abilities does {entity} use?",
            f"What arc is {entity} most important in?",
        ]

    def _answer_prefix(self, entity: str, entity_type: str, confidence: str) -> str:
        if confidence == "high":
            return f"This looks like {entity}."
        if confidence == "medium":
            return f"This is probably {entity}."
        if confidence == "low":
            return f"My best guess is {entity}, but the image is not fully clear."
        if entity_type == "group":
            return f"This appears to be {entity}."
        return f"This looks like {entity}."


vision_service = VisionService()
