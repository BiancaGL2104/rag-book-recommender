# src/retriever/retriever.py

import numpy as np
from src.retriever.embedder import Embedder
from src.retriever.vector_store import VectorStore


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
    # INTERNAL RERANKING
    # -------------------------------------------

    def rerank_results(self, query, results):
        """
        Improve retrieval using:
        - similarity from FAISS
        - book rating
        - genre overlap with query tokens
        """

        q_tokens = set(query.lower().split())

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

            # Final weighted score
            final_score = (
                0.70 * similarity +
                0.20 * rating_norm +
                0.10 * overlap_norm
            )

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
