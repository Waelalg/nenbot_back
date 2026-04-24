from openai import OpenAI
from app.config import GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL


def get_client() -> OpenAI:
    if not GROQ_API_KEY:
        raise ValueError("Missing GROQ_API_KEY in .env")
    return OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)


def generate_answer(messages: list[dict[str, str]]) -> str:
    client = get_client()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=700,
    )
    return (response.choices[0].message.content or "").strip()
