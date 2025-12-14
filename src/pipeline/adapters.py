# src/rag_pipeline/adapters.py

from typing import List, Dict, Any


def _normalize_genres(genres: Any) -> str:
    """
    Normalize genres into a comma-separated string.
    """
    if not genres:
        return ""
    if isinstance(genres, list):
        return ", ".join(str(g).strip() for g in genres if g)
    return str(genres).strip()


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def retrieve_books_for_llm(raw_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert FAISS retriever output into a clean, prompt-ready format
    for the RAG generator.

    This function:
    - normalizes metadata
    - avoids leaking internal scores to the LLM
    - guarantees stable keys for prompt formatting
    """
    books: List[Dict[str, Any]] = []

    for r in raw_results or []:
        metadata = r.get("metadata") or {}

        title = metadata.get("title") or metadata.get("Title") or "Unknown title"
        author = metadata.get("author") or metadata.get("Author") or "Unknown author"

        rating = (
            metadata.get("rating")
            or metadata.get("average_rating")
        )

        books.append({
            "Title": str(title),
            "Author": str(author),
            "genres": _normalize_genres(metadata.get("genres")),
            "average_rating": _safe_float(rating),
            "year": metadata.get("year"),
            "publisher": metadata.get("publisher"),
        })

    return books
