from __future__ import annotations

import logging
from dataclasses import dataclass
import re

from ..config import MAX_CONTEXT_CHUNKS
from ..data.hxh_aliases import enrich_retrieval_query
from ..rag.embeddings import embed_query
from ..rag.vector_store import count_documents, query_documents

logger = logging.getLogger(__name__)
GENERIC_TOKENS = {"hunter", "x", "arc", "family", "group", "city", "topic", "character", "characters"}


def _ranking_tokens(query: str, detected_entities: list[str]) -> set[str]:
    tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", f"{query} {' '.join(detected_entities)}".lower())
        if len(token) >= 4 and token not in GENERIC_TOKENS
    }
    return tokens


@dataclass
class RetrievalResult:
    context: str
    sources: list[str]
    best_distance: float | None = None

    @property
    def has_relevant_context(self) -> bool:
        return bool(self.context) and (self.best_distance is None or self.best_distance <= 1.15)


class RetrievalService:
    def build_context(
        self,
        query: str,
        detected_entities: list[str] | None = None,
        question_type: str | None = None,
    ) -> RetrievalResult:
        detected_entities = detected_entities or []
        retrieval_query = enrich_retrieval_query(query, detected_entities, question_type)
        try:
            if count_documents() == 0:
                return RetrievalResult(context="", sources=[], best_distance=None)
            embedding = embed_query(retrieval_query)
            results = query_documents(embedding, n_results=max(MAX_CONTEXT_CHUNKS * 2, 8))
        except Exception as exc:
            logger.exception("Retrieval failed")
            raise RuntimeError("Hunter x Hunter retrieval failed. Re-run ingestion and check ChromaDB/embeddings.") from exc

        docs = results.get("documents", [[]])[0] or []
        metadatas = results.get("metadatas", [[]])[0] or []
        distances = results.get("distances", [[]])[0] or []
        context_parts: list[str] = []
        sources: list[str] = []
        ranked_rows = []
        ranking_tokens = _ranking_tokens(query, detected_entities)
        for index, (doc, meta) in enumerate(zip(docs, metadatas)):
            haystack = f"{doc} {meta or {}}".lower()
            meta_haystack = (
                f"{(meta or {}).get('source', '')} {(meta or {}).get('topic', '')} {(meta or {}).get('section', '')}"
            ).lower()
            bonus = sum(1 for entity in detected_entities if entity.lower() in haystack)
            bonus += sum(0.35 for token in ranking_tokens if token in meta_haystack)
            bonus += sum(0.12 for token in ranking_tokens if token in haystack)
            if question_type == "arc_summary" and ("arc" in meta_haystack or "chimera" in meta_haystack):
                bonus += 0.35
            if "qa_examples" in meta_haystack:
                bonus -= 0.35
            distance = distances[index] if index < len(distances) else 99.0
            ranked_rows.append((distance - (bonus * 0.15), distance, doc, meta))

        seen_docs: set[str] = set()
        for _, distance, doc, meta in sorted(ranked_rows, key=lambda row: row[0]):
            fingerprint = doc[:240]
            if fingerprint in seen_docs:
                continue
            seen_docs.add(fingerprint)
            source = meta.get("source", "unknown") if meta else "unknown"
            topic = meta.get("topic", "unknown") if meta else "unknown"
            section = meta.get("section", "unknown") if meta else "unknown"
            sources.append(source)
            context_parts.append(f"[SOURCE: {source}; TOPIC: {topic}; SECTION: {section}]\n{doc}")
            if len(context_parts) >= MAX_CONTEXT_CHUNKS:
                break

        best_distance = min(distances) if distances else None
        return RetrievalResult(
            context="\n\n".join(context_parts),
            sources=sorted(set(sources)),
            best_distance=best_distance,
        )


retrieval_service = RetrievalService()

