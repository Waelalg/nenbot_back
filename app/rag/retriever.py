from app.config import MAX_CONTEXT_CHUNKS
from app.rag.embeddings import embed_query
from app.rag.vector_store import query_documents


def build_context(query: str) -> tuple[str, list[str]]:
    embedding = embed_query(query)
    results = query_documents(embedding, n_results=MAX_CONTEXT_CHUNKS)

    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    context_parts = []
    sources = []
    for doc, meta in zip(docs, metadatas):
        source = meta.get("source", "unknown") if meta else "unknown"
        sources.append(source)
        context_parts.append(f"[SOURCE: {source}]\n{doc}")

    return "\n\n".join(context_parts), sorted(set(sources))
