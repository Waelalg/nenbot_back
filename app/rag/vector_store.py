from typing import Any
from ..config import CHROMA_DIR, COLLECTION_NAME

_client = None
_collection = None


def get_client():
    global _client
    if _client is None:
        try:
            import chromadb
            from chromadb.config import Settings
        except ModuleNotFoundError as exc:
            raise RuntimeError("chromadb is not installed. Run `pip install -r requirements.txt`.") from exc
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def get_collection():
    global _collection
    if _collection is None:
        client = get_client()
        _collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return _collection


def reset_collection() -> None:
    global _collection
    client = get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    _collection = client.get_or_create_collection(name=COLLECTION_NAME)


def add_documents(ids: list[str], documents: list[str], embeddings: list[list[float]], metadatas: list[dict[str, Any]]) -> None:
    collection = get_collection()
    collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)


def query_documents(query_embedding: list[float], n_results: int = 4) -> dict[str, Any]:
    collection = get_collection()
    return collection.query(query_embeddings=[query_embedding], n_results=n_results)


def count_documents() -> int:
    collection = get_collection()
    return int(collection.count())


