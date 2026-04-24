from ..config import EMBEDDING_MODEL

_model = None


def get_embedding_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Run `pip install -r requirements.txt` in a supported Python environment."
            ) from exc
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    model = get_embedding_model()
    vector = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
    return vector.tolist()


