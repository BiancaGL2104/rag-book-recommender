# src/service/book_recommendation_service.py

from __future__ import annotations

from typing import Dict, Any, Optional, List
from collections import Counter

from src.pipeline.rag_pipeline import RAGPipeline
from src.generator.rag_generator import (
    RAGGenerator,
    LLMTimeoutError,
    LLMResponseFormatError,
)
from src.retriever.retriever import Retriever


class BookRecommendationService:
    """
    High-level facade used by the Streamlit UI.

    Responsibilities:
    - Hold singleton instances of Retriever, Generator, and RAGPipeline
      so heavy objects (FAISS index, embeddings, LLM client) are created only once.
    - Provide a simple .recommend(...) API for the UI.
    - Track lightweight recommendation statistics for the Analytics page.
    """

    _retriever: Optional[Retriever] = None
    _vector_store: Any = None
    _generator: Optional[RAGGenerator] = None
    _pipeline: Optional[RAGPipeline] = None

    def __init__(self):
        if BookRecommendationService._retriever is None:
            BookRecommendationService._retriever = Retriever(
                index_path="models/faiss_index.bin",
                metadata_path="models/metadata.pkl",
            )
        self.retriever: Retriever = BookRecommendationService._retriever

        if BookRecommendationService._vector_store is None:
            BookRecommendationService._vector_store = self._detect_vector_store(self.retriever)
        self.vector_store = BookRecommendationService._vector_store

        if BookRecommendationService._generator is None:
            BookRecommendationService._generator = RAGGenerator(model="llama3")
        self.generator: RAGGenerator = BookRecommendationService._generator

        if BookRecommendationService._pipeline is None:
            BookRecommendationService._pipeline = RAGPipeline(
                retriever=self.retriever,
                generator=self.generator,
            )
        self.pipeline: RAGPipeline = BookRecommendationService._pipeline

        self._recommend_counts: Counter[str] = Counter()

    def _detect_vector_store(self, retriever: Optional[Retriever] = None):
        r = retriever or self.retriever
        if hasattr(r, "vector_store"):
            return getattr(r, "vector_store")
        if hasattr(r, "vs"):
            return getattr(r, "vs")
        if hasattr(r, "store"):
            return getattr(r, "store")
        return None

    def _extract_title(self, book: Dict[str, Any]) -> str:
        """
        Robustly extract a title from either:
        - {"Title": "..."} (LLM-ready dicts)
        - {"title": "..."}
        - {"metadata": {"title": "..."}} (raw retrieval)
        """
        if not isinstance(book, dict):
            return ""

        meta = book.get("metadata", book)
        title = meta.get("title") or meta.get("Title") or book.get("Title") or book.get("title") or ""
        return str(title).strip()

    def _safe_retrieval(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Fallback used if LLM fails: return retriever candidates as LLM-ready dicts
        (same shape as pipeline returns in 'retrieved_books').
        """
        try:
            # Retriever.retrieve returns raw results: [{"metadata":..., "distance":..., "score":...}, ...]
            raw = self.retriever.retrieve(query, k=top_k, rerank=True)

            # Convert into the same structure your pipeline uses downstream
            # (Title/Author/genres/average_rating/year/publisher)
            books: List[Dict[str, Any]] = []
            for r in raw:
                m = r.get("metadata", {})
                books.append({
                    "Title": m.get("title") or m.get("Title") or "Unknown title",
                    "Author": m.get("author") or m.get("Author") or "Unknown author",
                    "genres": m.get("genres", ""),
                    "average_rating": float(m.get("rating") or m.get("average_rating") or 0.0),
                    "year": m.get("year"),
                    "publisher": m.get("publisher"),
                    "score": r.get("score"),
                })
            return books
        except Exception:
            return []

    def _update_recommend_counts(self, books: List[Dict[str, Any]]) -> None:
        for b in books:
            title = self._extract_title(b)
            if title:
                self._recommend_counts[title] += 1

    # ------------------- Public API -------------------

    def recommend(
        self,
        query: str,
        style: Optional[str] = None,
        use_mood: bool = True,
        explain: bool = False,
        second_opinion: bool = False,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Main entry point used by the Streamlit UI.
        """
        mood_override = None if use_mood else "neutral"
        style_value = style or "default"

        try:
            result = self.pipeline.run(
                query=query,
                style=style_value,
                history=history,
                mood=mood_override,
                explain=explain,
                second_opinion=second_opinion,
            )

        except LLMTimeoutError:
            fallback_books = self._safe_retrieval(query, top_k=5)
            result = {
                "answer": (
                    "The language model took too long to respond, so I’m showing "
                    "the closest matches retrieved directly from the catalog."
                ),
                "retrieved_books": fallback_books,
                "recommended_books": [],
                "context": "",
                "query": query,
            }

        except LLMResponseFormatError:
            fallback_books = self._safe_retrieval(query, top_k=5)
            result = {
                "answer": (
                    "I had trouble interpreting the model’s answer this time. "
                    "Here are candidate books retrieved directly from the catalog."
                ),
                "retrieved_books": fallback_books,
                "recommended_books": [],
                "context": "",
                "query": query,
            }

        # Track stats (works for both Title/title variants)
        recommended_books = result.get("recommended_books") or []
        self._update_recommend_counts(recommended_books)

        return result

    # ------------------- Analytics helpers -------------------

    def get_recommendation_stats(self) -> Dict[str, int]:
        return dict(self._recommend_counts)

    def get_all_titles(self) -> List[str]:
        """
        Return all titles from the vector store metadata (for graph dropdown, etc.).
        """
        if not self.vector_store:
            return []
        metas = self.vector_store.get_all_metadata()
        titles: List[str] = []
        for m in metas:
            t = (m.get("title") or m.get("Title") or "").strip()
            if t:
                titles.append(t)
        return titles

    def get_all_metadata(self) -> List[Dict[str, Any]]:
        if not self.vector_store:
            return []
        return self.vector_store.get_all_metadata()

    def get_vector_store(self):
        return self.vector_store
