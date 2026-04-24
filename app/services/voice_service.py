from __future__ import annotations

import logging
from typing import Final

import requests

from ..config import (
    GROQ_API_KEY,
    GROQ_BASE_URL,
    VOICE_STT_MODEL,
    VOICE_TTS_FORMAT,
    VOICE_TTS_MODEL,
    VOICE_TTS_VOICE,
)

logger = logging.getLogger(__name__)

_TRANSCRIBE_ENDPOINT: Final[str] = "/audio/transcriptions"
_SPEECH_ENDPOINT: Final[str] = "/audio/speech"
_CONTENT_TYPE_MAP: Final[dict[str, str]] = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "flac": "audio/flac",
}
_MIME_EXTENSION_MAP: Final[dict[str, str]] = {
    "audio/webm": ".webm",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/mp4": ".m4a",
    "audio/aac": ".aac",
    "audio/ogg": ".ogg",
}
_TRANSCRIPTION_PROMPT: Final[str] = (
    "Hunter x Hunter names and terms: Hunter x Hunter, Gon Freecss, Killua Zoldyck, Kurapika, "
    "Leorio Paradinight, Hisoka Morow, Chrollo Lucilfer, Phantom Troupe, Nen, Greed Island, "
    "Chimera Ant, Zoldyck Family, Meruem, Netero."
)


class VoiceService:
    def _api_url(self, suffix: str) -> str:
        return f"{GROQ_BASE_URL.rstrip('/')}{suffix}"

    def _headers(self, *, json_request: bool = False) -> dict[str, str]:
        if not GROQ_API_KEY:
            raise RuntimeError("Missing GROQ_API_KEY. Add it to .env before using voice features.")
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
        if json_request:
            headers["Content-Type"] = "application/json"
        return headers

    def transcribe_audio(self, audio_bytes: bytes, content_type: str | None = None, language: str | None = None) -> str:
        if not audio_bytes:
            raise RuntimeError("No audio was received for transcription.")

        mime_type = (content_type or "audio/webm").split(";")[0].strip().lower()
        extension = _MIME_EXTENSION_MAP.get(mime_type, ".webm")
        files = {"file": (f"voice{extension}", audio_bytes, mime_type)}
        data: dict[str, str] = {
            "model": VOICE_STT_MODEL,
            "response_format": "json",
            "temperature": "0",
            "prompt": _TRANSCRIPTION_PROMPT,
        }
        if language:
            data["language"] = language

        try:
            response = requests.post(
                self._api_url(_TRANSCRIBE_ENDPOINT),
                headers=self._headers(),
                data=data,
                files=files,
                timeout=90,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            message = self._friendly_error(exc, "NENBOT could not transcribe the recorded audio.")
            logger.warning("Groq transcription failed: %s", message)
            raise RuntimeError(message) from exc

        payload = response.json()
        transcript = str(payload.get("text") or "").strip()
        if not transcript:
            raise RuntimeError("The transcription result was empty.")
        return transcript

    def synthesize_speech(self, text: str) -> tuple[bytes, str]:
        prompt = text.strip()
        if not prompt:
            raise RuntimeError("No text was provided for speech generation.")
        if len(prompt) > 200:
            raise RuntimeError("Groq text-to-speech accepts up to 200 characters per request. Split the text into shorter segments before calling /voice/speak.")

        try:
            response = requests.post(
                self._api_url(_SPEECH_ENDPOINT),
                headers=self._headers(json_request=True),
                json={
                    "model": VOICE_TTS_MODEL,
                    "voice": VOICE_TTS_VOICE,
                    "input": prompt,
                    "response_format": VOICE_TTS_FORMAT,
                },
                timeout=90,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            message = self._friendly_error(exc, "NENBOT could not generate speech for this answer.")
            logger.warning("Groq speech generation failed: %s", message)
            raise RuntimeError(message) from exc

        media_type = _CONTENT_TYPE_MAP.get(VOICE_TTS_FORMAT, "audio/wav")
        return response.content, media_type

    def _friendly_error(self, exc: requests.RequestException, fallback: str) -> str:
        response = exc.response
        if response is None:
            return fallback

        try:
            payload = response.json()
        except ValueError:
            return fallback

        error_data = payload.get("error") or {}
        error_code = str(error_data.get("code") or "")
        error_message = str(error_data.get("message") or "").strip()

        if error_code == "model_terms_required":
            return "Groq text-to-speech is not enabled for this account yet. Accept the model terms in the Groq console or use browser voice output."
        if error_message:
            return error_message
        return fallback


voice_service = VoiceService()


