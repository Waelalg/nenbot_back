from __future__ import annotations

from dataclasses import dataclass
import re


FIELD_LABELS = {
    "type",
    "aliases",
    "related",
    "summary",
    "key facts",
    "main characters",
    "conflict",
    "important concepts",
    "important events",
    "why it matters",
    "user",
    "nen category",
    "effect",
    "limitation",
    "arc relevance",
}

BLOCK_PATTERN = re.compile(
    r"\[SOURCE: (?P<source>[^;]+); TOPIC: (?P<topic>[^;]+); SECTION: (?P<section>[^\]]+)\]\n(?P<text>.*?)(?=\n\n\[SOURCE: |\Z)",
    re.S,
)
GENERIC_SCORE_TOKENS = {"hunter", "arc", "family", "group", "city", "topic", "character", "characters"}


@dataclass(frozen=True)
class ContextBlock:
    source: str
    topic: str
    section: str
    text: str


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        cleaned = re.sub(r"\s+", " ", item).strip(" -:\n\t")
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(cleaned)
    return ordered


def _parse_blocks(context: str) -> list[ContextBlock]:
    blocks: list[ContextBlock] = []
    for match in BLOCK_PATTERN.finditer(context or ""):
        blocks.append(
            ContextBlock(
                source=match.group("source").strip(),
                topic=match.group("topic").strip(),
                section=match.group("section").strip(),
                text=match.group("text").strip(),
            )
        )
    return blocks


def _ranking_tokens(query: str, detected_entities: list[str]) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", f"{query} {' '.join(detected_entities)}".lower())
        if len(token) >= 4 and token not in GENERIC_SCORE_TOKENS
    }


def _blocks_for_arc_summary(blocks: list[ContextBlock], query: str, detected_entities: list[str]) -> list[ContextBlock]:
    tokens = _ranking_tokens(query, detected_entities)
    selected = [
        block
        for block in blocks
        if "arc" in block.section.lower()
        or any(token in f"{block.source} {block.topic} {block.section}".lower() for token in tokens)
    ]
    return selected or blocks


def _score_block(block: ContextBlock, query: str, detected_entities: list[str]) -> float:
    haystack = f"{block.source} {block.topic} {block.section} {block.text}".lower()
    meta_haystack = f"{block.source} {block.topic} {block.section}".lower()
    score = 0.0
    for entity in detected_entities:
        entity_lower = entity.lower()
        entity_base = entity_lower.replace(" arc", "")
        if entity_lower in meta_haystack:
            score += 7.0
        elif entity_lower in haystack:
            score += 4.0
        if entity_base in meta_haystack:
            score += 2.0
    for token in _ranking_tokens(query, detected_entities):
        if token in meta_haystack:
            score += 1.5
        elif token in haystack:
            score += 0.4
    if "qa_examples" in block.source.lower():
        score -= 2.0
    return score


def _field_value(text: str, label: str) -> list[str]:
    values: list[str] = []
    capture = False
    label_lower = label.lower()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if capture and values:
                break
            continue
        if line.startswith("#"):
            capture = False
            continue
        if capture:
            head = line.rstrip(":").lower()
            if head in FIELD_LABELS or line.startswith("## "):
                break
            values.append(line)
            continue
        if line.lower() == f"{label_lower}:":
            capture = True
            continue
        if line.lower().startswith(f"{label_lower}:"):
            values.append(line.split(":", 1)[1].strip())
            break
    return _unique(values)


def _first_descriptive_text(text: str) -> str:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("- "):
            continue
        if ":" in line and line.split(":", 1)[0].strip().lower() in FIELD_LABELS:
            continue
        return line
    return ""


def _headings_and_notes(blocks: list[ContextBlock]) -> list[str]:
    items: list[str] = []
    for block in blocks:
        section_lower = block.section.lower()
        if block.section not in {"Document", "Untitled"} and block.section and not section_lower.startswith(("q:", "a:", "entity:")):
            items.append(block.section)
        lines = block.text.splitlines()
        for index, raw_line in enumerate(lines):
            line = raw_line.strip()
            if not line.startswith("## "):
                continue
            title = line[3:].strip()
            if title.lower().startswith(("q:", "a:", "entity:")):
                continue
            snippet = ""
            for next_line in lines[index + 1 :]:
                candidate = next_line.strip()
                if not candidate or candidate.startswith("#") or candidate.startswith("- "):
                    continue
                if ":" in candidate and candidate.split(":", 1)[0].strip().lower() in FIELD_LABELS:
                    continue
                snippet = candidate
                break
            items.append(f"{title}: {snippet}".strip(": "))
    return _unique(items)


def _collect_bullets(blocks: list[ContextBlock]) -> list[str]:
    bullets: list[str] = []
    for block in blocks:
        for raw_line in block.text.splitlines():
            line = raw_line.strip()
            if line.startswith("- "):
                bullets.append(line[2:].strip())
    return _unique(bullets)


def _collect_field(blocks: list[ContextBlock], label: str) -> list[str]:
    collected: list[str] = []
    for block in blocks:
        collected.extend(_field_value(block.text, label))
    return _unique(collected)


def _collect_category_list(blocks: list[ContextBlock]) -> list[str]:
    items: list[str] = []
    for block in blocks:
        for raw_line in block.text.splitlines():
            line = raw_line.strip()
            if line.startswith("## Category:"):
                items.append(line.split(":", 1)[1].strip())
    return _unique(items)


def _collect_arc_list(blocks: list[ContextBlock]) -> list[str]:
    items: list[str] = []
    for block in blocks:
        for raw_line in block.text.splitlines():
            line = raw_line.strip()
            if line.startswith("## Arc:"):
                items.append(line.split(":", 1)[1].strip())
    return _unique(items)


def _collect_member_list(blocks: list[ContextBlock]) -> list[str]:
    members: list[str] = []
    for block in blocks:
        for value in _field_value(block.text, "Main characters"):
            members.extend([item.strip() for item in value.split(",") if item.strip()])
        for note in _headings_and_notes([block]):
            lower = note.lower()
            if lower.startswith("major members:") or lower.startswith("important members include"):
                payload = note.split(":", 1)[1] if ":" in note else note
                members.extend([item.strip(" .") for item in payload.split(",") if item.strip()])
    return _unique(members)


def _entity_name(query: str, detected_entities: list[str], blocks: list[ContextBlock]) -> str:
    if detected_entities:
        return detected_entities[0]
    for block in blocks:
        line = next((item.strip() for item in block.text.splitlines() if item.strip().startswith("## ")), "")
        if line:
            return line.lstrip("# ").strip()
    return query.strip().rstrip("?") or "this Hunter x Hunter topic"


class FallbackAnswerService:
    def build_answer(
        self,
        *,
        query: str,
        question_type: str,
        retrieved_context: str,
        detected_entities: list[str] | None = None,
    ) -> str:
        detected_entities = detected_entities or []
        blocks = sorted(
            _parse_blocks(retrieved_context),
            key=lambda block: _score_block(block, query, detected_entities),
            reverse=True,
        )
        non_qa_blocks = [block for block in blocks if "qa_examples" not in block.source.lower()]
        if non_qa_blocks:
            blocks = non_qa_blocks
        if not blocks:
            return "I do not have enough Hunter x Hunter information in the local knowledge base to answer that."

        entity_name = _entity_name(query, detected_entities, blocks)
        summaries = _unique(
            [
                *[item for block in blocks for item in _field_value(block.text, "Summary")],
                *[item for block in blocks if (item := _first_descriptive_text(block.text))],
            ]
        )
        bullets = _collect_bullets(blocks)
        headings = _headings_and_notes(blocks)
        why_it_matters = _collect_field(blocks, "Why it matters")
        main_characters = _collect_field(blocks, "Main characters")
        users = _collect_field(blocks, "User")
        nen_categories = _collect_field(blocks, "Nen category")
        effects = _collect_field(blocks, "Effect")
        limits = _collect_field(blocks, "Limitation")

        if question_type == "list":
            query_lower = query.lower()
            if "nen" in query_lower and any(term in query_lower for term in ["type", "types", "category", "categories"]):
                items = _collect_category_list(blocks)
            elif "arc" in query_lower:
                items = _collect_arc_list(blocks)
            elif "member" in query_lower:
                items = _collect_member_list(blocks)
            else:
                items = bullets or headings
            items = (items or summaries)[:10]
            if not items:
                return "I found Hunter x Hunter context, but not a clean list for that question."
            return f"{entity_name}:\n" + "\n".join(f"{index + 1}. {item}" for index, item in enumerate(items))

        if question_type == "ability":
            parts: list[str] = []
            if summaries:
                parts.append(summaries[0])
            if users:
                parts.append(f"User: {users[0]}.")
            if nen_categories:
                parts.append(f"Nen category: {nen_categories[0]}.")
            if effects:
                parts.append(f"Effect: {effects[0]}.")
            if limits:
                parts.append(f"Limitation: {limits[0]}.")
            if len(headings) > 1:
                parts.append("Relevant details:\n" + "\n".join(f"- {item}" for item in headings[:4]))
            return "\n\n".join(parts).strip()

        if question_type == "arc_summary":
            arc_blocks = _blocks_for_arc_summary(blocks, query, detected_entities)
            arc_summaries = _unique(
                [
                    *[item for block in arc_blocks for item in _field_value(block.text, "Summary")],
                    *[item for block in arc_blocks if (item := _first_descriptive_text(block.text))],
                ]
            )
            arc_main_characters = _collect_field(arc_blocks, "Main characters")
            arc_bullets = _collect_bullets(arc_blocks)
            arc_headings = _headings_and_notes(arc_blocks)
            arc_why = _collect_field(arc_blocks, "Why it matters")
            parts = [arc_summaries[0] if arc_summaries else f"{entity_name} is covered in the local Hunter x Hunter knowledge base."]
            if arc_main_characters:
                parts.append(f"Main characters: {arc_main_characters[0]}")
            if arc_bullets:
                parts.append("Key points:\n" + "\n".join(f"- {item}" for item in arc_bullets[:6]))
            if arc_headings:
                parts.append("Important moments:\n" + "\n".join(f"- {item}" for item in arc_headings[:5]))
            if arc_why:
                parts.append(f"Why it matters: {arc_why[0]}")
            return "\n\n".join(parts).strip()

        if question_type == "relationship":
            parts = [summaries[0] if summaries else f"{entity_name} has a relationship entry in the local knowledge base."]
            points = bullets[:4] or headings[:4]
            if points:
                parts.append("Key points:\n" + "\n".join(f"- {item}" for item in points))
            return "\n\n".join(parts).strip()

        if question_type in {"simple_explanation", "detailed_explanation", "definition", "identity", "memory_followup"}:
            parts = [summaries[0] if summaries else f"{entity_name} is covered in the local Hunter x Hunter knowledge base."]
            if bullets:
                parts.append("Key points:\n" + "\n".join(f"- {item}" for item in bullets[:6]))
            elif headings:
                parts.append("Relevant details:\n" + "\n".join(f"- {item}" for item in headings[:5]))
            if why_it_matters and question_type in {"detailed_explanation", "definition", "identity"}:
                parts.append(f"Why it matters: {why_it_matters[0]}")
            return "\n\n".join(parts).strip()

        if question_type == "comparison":
            items = bullets[:6] or headings[:6] or summaries[:4]
            if not items:
                return f"I found Hunter x Hunter context for {entity_name}, but not enough local detail for a strong comparison."
            return f"Relevant comparison points for {entity_name}:\n" + "\n".join(f"- {item}" for item in items)

        intro = summaries[0] if summaries else f"{entity_name} is covered in the local Hunter x Hunter knowledge base."
        detail_points = bullets[:5] or headings[:5]
        if detail_points:
            return intro + "\n\nKey points:\n" + "\n".join(f"- {item}" for item in detail_points)
        return intro


fallback_answer_service = FallbackAnswerService()
