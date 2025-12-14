# src/retriever/utils.py

from __future__ import annotations
from typing import List, Dict, Any
import re


class SimilarBook:
    """
    Simple container for similar-book results.
    """
    def __init__(self, title: str, score: float, metadata: Dict[str, Any]):
        self.title = title
        self.score = score
        self.metadata = metadata


def _normalize_title(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip().casefold())


def get_similar_books_by_title(
    title: str,
    vector_store,
    embedder,
    k: int = 10,
) -> List[SimilarBook]:
    """
    Given a book title, find its k most similar books based on embeddings.

    Works with the custom FAISS VectorStore used in this project.

    Parameters
    ----------
    title : str
        Book title to search for.
    vector_store : VectorStore
        Instance of src.retriever.vector_store.VectorStore
    embedder : Embedder
        Instance of src.retriever.embedder.Embedder
    k : int
        Number of similar books to retrieve (including itself).

    Returns
    -------
    List[SimilarBook]
    """
    if not title:
        return []

    target_meta = None
    title_norm = _normalize_title(title)

    for m in vector_store.metadata:
        m_title = _normalize_title(m.get("title") or m.get("Title"))
        if m_title == title_norm:
            target_meta = m
            break

    if target_meta is None:
        raise ValueError(f"Book with title '{title}' not found in vector store metadata.")

    parts = [target_meta.get("title") or target_meta.get("Title") or ""]
    desc = (
        target_meta.get("description")
        or target_meta.get("retrieval_text")
        or ""
    )
    if desc:
        parts.append(desc[:400])

    base_text = ". ".join(p for p in parts if p)

    query_vec = embedder.encode(base_text)
    if query_vec.size == 0:
        return []

    results = vector_store.search(query_vec[0], k=k)

    similar: List[SimilarBook] = []
    for r in results:
        meta = r.get("metadata", {})
        dist = r.get("distance")

        try:
            score = max(0.0, 1.0 - float(dist))
        except (TypeError, ValueError):
            score = 0.0

        similar.append(
            SimilarBook(
                title=meta.get("title") or meta.get("Title") or "Unknown title",
                score=score,
                metadata=meta,
            )
        )

    similar.sort(key=lambda sb: sb.score, reverse=True)
    return similar
