SYSTEM_PROMPT = """You are NENBOT, a Hunter x Hunter-only AI assistant.

Allowed topics:
1. Hunter x Hunter universe, characters, Nen, arcs, factions, powers, events, and lore
2. Project team information: full names, academic level, age, university name, and field of study
3. Very short usage help about NENBOT itself

Rules:
- Never answer questions outside those allowed topics.
- If a question is unrelated, politely refuse and say you are specialized only in Hunter x Hunter and team information.
- Prefer the retrieved context and structured team context.
- If the answer is not available in context, say you do not have enough information.
- Do not invent facts.
- Keep the answer clear, direct, and helpful.
- If the user asks a follow-up question, use the conversation memory to resolve pronouns or references.
"""


def build_messages(user_question: str, history: list[dict[str, str]], intent: str, retrieved_context: str = "", team_context: str = "") -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    if history:
        messages.append({
            "role": "system",
            "content": "Conversation memory (latest interactions):\n" + "\n".join(
                [f"{m['role'].upper()}: {m['content']}" for m in history]
            )
        })

    if retrieved_context:
        messages.append({"role": "system", "content": f"Retrieved Hunter x Hunter context:\n{retrieved_context}"})

    if team_context:
        messages.append({"role": "system", "content": f"Structured team data:\n{team_context}"})

    messages.append({
        "role": "system",
        "content": f"Detected intent: {intent}. Stay inside scope. Refuse if out of scope."
    })

    messages.append({"role": "user", "content": user_question})
    return messages

