# src/retriever/retriever.py

import re
import numpy as np
from src.retriever.embedder import Embedder
from src.retriever.vector_store import VectorStore
from src.utils.filter_parser import parse_filters

# -------------------------------------------------
# THEME & TONE KEYWORDS (Option B: general, reusable)
# -------------------------------------------------

THEME_KEYWORDS = {
    "academy": [
        "academ", "school", "university", "college", "campus", "professor", "library"
    ],
    "mystery": [
        "mystery", "detective", "crime", "whodunit", "investigation"
    ],
    "politics": [
        "political", "court", "rebellion", "kingdom", "power", "conspiracy"
    ],
    "romance": [
        "romance", "love", "relationship", "marriage", "wedding"
    ],
    "fantasy": [
        "magic", "fantasy", "kingdom", "wizard", "fae", "dragon"
    ],
    "sci_fi": [
        "space", "alien", "future", "cyberpunk", "starship", "ai", "robot"
    ],
    "post_apoc": [
        "post-apocalyptic", "apocalypse", "wasteland", "outbreak", "survival"
    ],
    "found_family": [
        "found family", "ragtag", "band of misfits", "crew"
    ],
    "historical": [
        "historical", "19th century", "victorian", "regency", "wwii"
    ],
}

TONE_KEYWORDS = {
    "cozy": [
        "cozy", "wholesome", "heartwarming", "comforting", "light", "feel-good"
    ],
    "dark": [
        "dark", "gritty", "grim", "bleak", "ominous", "twisted"
    ],
    "atmospheric": [
        "atmospheric", "moody", "haunting", "gothic"
    ],
    "fast_paced": [
        "fast-paced", "page-turner", "non-stop", "action-packed"
    ],
    "slow_burn": [
        "slow burn", "slow-burn", "gradual", "character-driven"
    ],
}


class Retriever:
    """
    Retriever = embeddings + FAISS search + internal reranking

    Returns a list of dicts:
    {
        "metadata": {... original book metadata ...},
        "distance": float,
        "similarity": float,
        "score": float
    }
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

    # -------------------------------------------
    # EMBEDDING
    # -------------------------------------------

    def embed_query(self, query: str):
        """Embed the user query into a dense vector using the sentence transformer."""
        return self.embedder.encode(query)[0]

    # -------------------------------------------
    # MAIN RETRIEVAL ENTRY POINT
    # -------------------------------------------

    def retrieve(self, query: str, k=None, rerank=True):
        """
        Retrieve top-k documents using FAISS + optional reranking.
        Returns a list of dicts with:
            - metadata
            - distance
            - similarity
            - score (after reranking)
        """

        # If caller did not specify k, use default
        if k is None:
            k = self.k

        # 1) Embed query
        q_vec = self.embed_query(query)

        # 2) FAISS top-k search
        results = self.vector_store.search(q_vec, k)

        # 3) Rerank internally
        if rerank:
            results = self.rerank_results(query, results)

        return results

    # -------------------------------------------
    # HELPER: THEME & TONE SCORING
    # -------------------------------------------

    def _compute_theme_score(self, query: str, text: str) -> float:
        """
        Count how many themes appear both in the query and in the book text.
        Returns a normalized score in [0, 1].
        """
        q = query.lower()
        t = text.lower()
        hits = 0

        for _theme, words in THEME_KEYWORDS.items():
            q_hit = any(w in q for w in words)
            t_hit = any(w in t for w in words)
            if q_hit and t_hit:
                hits += 1

        # Normalize: cap at 1.0 (3+ overlapping themes all count as 1)
        if hits == 0:
            return 0.0
        return min(hits / 3.0, 1.0)

    def _compute_tone_alignment(self, query: str, text: str) -> float:
        """
        Measure how well the tone in the text matches the tone implied by the query.
        Returns a normalized score in [0, 1].
        """
        q = query.lower()
        t = text.lower()
        hits = 0

        for _tone, words in TONE_KEYWORDS.items():
            q_hit = any(w in q for w in words)
            t_hit = any(w in t for w in words)
            if q_hit and t_hit:
                hits += 1

        if hits == 0:
            return 0.0
        return min(hits / 2.0, 1.0)

    def _is_tone_mismatch(self, query: str, text: str) -> bool:
        """
        Detect strong tone mismatches:
        - query asks for cozy/light but text looks dark/gritty
        - can be extended with more rules later
        """
        q = query.lower()
        t = text.lower()

        q_cozy = any(w in q for w in TONE_KEYWORDS["cozy"])
        t_dark = any(w in t for w in TONE_KEYWORDS["dark"])

        # Example rule: user wants cozy/light but book description is dark
        if q_cozy and t_dark:
            return True

        return False

    # -------------------------------------------
    # INTERNAL RERANKING
    # -------------------------------------------

    def rerank_results(self, query, results):
        """
        Improve retrieval using:
        - similarity from FAISS
        - book rating
        - genre overlap with query tokens
        - theme-based overlap (setting, plot type, etc.)
        - tone alignment (cozy/dark/atmospheric etc.)
        - soft filtering based on simple numeric constraints (pages, rating)
        """

        q_tokens = set(query.lower().split())
        filters = parse_filters(query)  # e.g. {"max_pages": 300, "min_rating": 4.0}

        reranked = []

        for r in results:
            meta = r["metadata"]
            distance = r["distance"]

            # Convert FAISS L2 distance to similarity
            similarity = 1 / (1 + distance)

            # Rating
            rating = meta.get("rating", 0) or 0
            rating_norm = rating / 5  # normalize to 0–1

            # Genres: stored as a comma-separated string
            genres_str = meta.get("genres", "").lower()
            genre_tokens = set(g.strip() for g in genres_str.replace(",", " ").split())
            overlap = len(q_tokens & genre_tokens)
            overlap_norm = overlap / (len(q_tokens) + 1)

            # Book text used for theme & tone:
            # fallback from retrieval_text to description if needed
            text = (
                meta.get("retrieval_text")
                or meta.get("description")
                or ""
            )
            text = text.lower()

            # Theme & tone scores
            theme_norm = self._compute_theme_score(query, text)      # [0,1]
            tone_align = self._compute_tone_alignment(query, text)   # [0,1]

            # Base weighted score
            final_score = (
                0.60 * similarity +
                0.15 * rating_norm +
                0.10 * overlap_norm +
                0.10 * theme_norm +
                0.05 * tone_align
            )

            # ----------------------------
            # Soft penalties from filters
            # ----------------------------
            # Pages: try to read from common keys, default to None if missing
            pages = meta.get("num_pages") or meta.get("pages") or None
            if isinstance(pages, str):
                try:
                    pages = int(pages)
                except ValueError:
                    pages = None

            # min_rating
            min_rating = filters.get("min_rating")
            if min_rating is not None and rating and rating < min_rating:
                # Penalize but don't completely remove
                final_score *= 0.6

            # max_pages
            max_pages = filters.get("max_pages")
            if max_pages is not None and pages is not None and pages > max_pages:
                final_score *= 0.6

            # min_pages
            min_pages = filters.get("min_pages")
            if min_pages is not None and pages is not None and pages < min_pages:
                final_score *= 0.6

            # Optional penalty for strong tone mismatch
            if self._is_tone_mismatch(query, text):
                final_score *= 0.8

            reranked.append({
                "metadata": meta,
                "distance": distance,
                "similarity": similarity,
                "score": final_score,
            })

        # Highest score first
        reranked.sort(key=lambda x: x["score"], reverse=True)
        return reranked

    # -------------------------------------------
    # OPTIONAL: FORMAT FOR LLM (external use only)
    # -------------------------------------------

    def format_for_llm(self, retrieved_docs):
        """
        Convert retrieved docs into a clean text block for prompting.
        Not used by RAG pipeline, but useful for debugging/notebooks.
        """
        lines = []
        for i, r in enumerate(retrieved_docs, 1):
            m = r["metadata"]
            line = (
                f"{i}. Title: {m.get('title', 'Unknown')}\n"
                f"   Author: {m.get('author', 'Unknown')}\n"
                f"   Genres: {m.get('genres', 'Unknown')}\n"
                f"   Rating: {m.get('rating', 'N/A')}\n"
            )
            lines.append(line)
        return "\n".join(lines)
