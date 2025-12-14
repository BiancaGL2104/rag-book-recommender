# src/retriever/retriever.py

import re
from typing import List, Dict, Any

from src.retriever.embedder import Embedder
from src.retriever.vector_store import VectorStore
from src.utils.filter_parser import parse_filters


THEME_KEYWORDS = {
    "academy": ["academ", "school", "university", "college", "campus", "professor", "library"],
    "mystery": ["mystery", "detective", "crime", "whodunit", "investigation"],
    "politics": ["political", "court", "rebellion", "kingdom", "power", "conspiracy"],
    "romance": ["romance", "love", "relationship", "marriage", "wedding"],
    "fantasy": ["magic", "fantasy", "kingdom", "wizard", "fae", "dragon"],
    "sci_fi": ["space", "alien", "future", "cyberpunk", "starship", "ai", "robot"],
    "post_apoc": ["post-apocalyptic", "apocalypse", "wasteland", "outbreak", "survival"],
    "found_family": ["found family", "ragtag", "band of misfits", "crew"],
    "historical": ["historical", "victorian", "regency", "wwii"],
}

TONE_KEYWORDS = {
    "cozy": ["cozy", "wholesome", "heartwarming", "comforting", "light", "feel-good"],
    "dark": ["dark", "gritty", "grim", "bleak", "ominous", "twisted"],
    "atmospheric": ["atmospheric", "moody", "haunting", "gothic"],
    "fast_paced": ["fast-paced", "page-turner", "non-stop", "action-packed"],
    "slow_burn": ["slow burn", "slow-burn", "gradual", "character-driven"],
}


class Retriever:
    """
    Retriever = embeddings + FAISS search + heuristic reranking.
    """

    def __init__(
        self,
        index_path="models/faiss_index.bin",
        metadata_path="models/metadata.pkl",
        model_name="all-MiniLM-L6-v2",
        k=5,
    ):
        self.k = k
        self.embedder = Embedder(model_name=model_name)
        self.vector_store = VectorStore.load(index_path, meta_path=metadata_path)

    def embed_query(self, query: str):
        return self.embedder.encode(query)[0]

    def retrieve(self, query: str, k: int | None = None, rerank: bool = True):
        if not query:
            return []

        if k is None:
            k = self.k

        q_vec = self.embed_query(query)
        results = self.vector_store.search(q_vec, k)

        if rerank:
            results = self.rerank_results(query, results)

        return results

    def _compute_theme_score(self, query: str, text: str) -> float:
        q, t = query.lower(), text.lower()
        hits = sum(
            any(w in q for w in words) and any(w in t for w in words)
            for words in THEME_KEYWORDS.values()
        )
        return min(hits / 3.0, 1.0)

    def _compute_tone_alignment(self, query: str, text: str) -> float:
        q, t = query.lower(), text.lower()
        hits = sum(
            any(w in q for w in words) and any(w in t for w in words)
            for words in TONE_KEYWORDS.values()
        )
        return min(hits / 2.0, 1.0)

    def _is_tone_mismatch(self, query: str, text: str) -> bool:
        q, t = query.lower(), text.lower()
        return (
            any(w in q for w in TONE_KEYWORDS["cozy"])
            and any(w in t for w in TONE_KEYWORDS["dark"])
        )

    def rerank_results(self, query: str, results: List[Dict[str, Any]]):
        q_tokens = set(query.lower().split())
        filters = parse_filters(query)
        reranked = []

        for r in results:
            meta = r.get("metadata", {})
            distance = float(r.get("distance", 1.0))

            similarity = max(0.0, 1.0 - distance)

            rating = (
                meta.get("rating")
                or meta.get("average_rating")
                or 0.0
            )
            rating_norm = min(float(rating) / 5.0, 1.0)

            genres = meta.get("genres", "")
            if isinstance(genres, list):
                genre_tokens = set(g.lower() for g in genres)
            else:
                genre_tokens = set(genres.lower().replace(",", " ").split())

            overlap = len(q_tokens & genre_tokens)
            overlap_norm = overlap / (len(q_tokens) + 1)

            text = (
                meta.get("retrieval_text")
                or meta.get("description")
                or ""
            ).lower()

            theme_norm = self._compute_theme_score(query, text)
            tone_align = self._compute_tone_alignment(query, text)

            score = (
                0.60 * similarity +
                0.15 * rating_norm +
                0.10 * overlap_norm +
                0.10 * theme_norm +
                0.05 * tone_align
            )

            pages = meta.get("num_pages") or meta.get("pages")
            try:
                pages = int(pages) if pages is not None else None
            except ValueError:
                pages = None

            if filters.get("min_rating") and rating < filters["min_rating"]:
                score *= 0.6
            if filters.get("max_pages") and pages and pages > filters["max_pages"]:
                score *= 0.6
            if filters.get("min_pages") and pages and pages < filters["min_pages"]:
                score *= 0.6
            if self._is_tone_mismatch(query, text):
                score *= 0.8

            reranked.append({
                "metadata": meta,
                "distance": distance,
                "similarity": similarity,
                "score": max(score, 0.0),
            })

        reranked.sort(key=lambda x: x["score"], reverse=True)
        return reranked
