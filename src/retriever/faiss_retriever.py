# src/retriever/faiss_retriever.py

from __future__ import annotations

from typing import List, Dict, Any
import numpy as np

from src.retriever.embedder import Embedder
from src.retriever.vector_store import VectorStore


class FAISSRetriever:
    """
    Retrieval component based on FAISS + SentenceTransformer embeddings.

    Responsibilities:
    - load FAISS index and metadata
    - embed user queries
    - return top-k nearest neighbors with metadata
    """

    def __init__(
        self,
        index_path: str = "models/faiss_index.bin",
        metadata_path: str = "models/metadata.pkl",
        embedder: Embedder | None = None,
    ):
        """
        Parameters
        ----------
        index_path : str
            Path to FAISS index file.
        metadata_path : str
            Path to pickled metadata list.
        embedder : Embedder, optional
            Custom embedder instance (otherwise default Embedder is used).
        """
        self.embedder = embedder or Embedder()
        self.vector_store = VectorStore.load(
            index_path=index_path,
            meta_path=metadata_path,
        )

    def retrieve(self, query: str, k: int = 6) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most similar documents for a query.

        Parameters
        ----------
        query : str
            User query text.
        k : int
            Number of results to retrieve.

        Returns
        -------
        List[Dict[str, Any]]
            Each item has:
            {
                "metadata": {...},
                "similarity": float
            }
        """
        if not query or not query.strip():
            return []

        query_vec = self.embedder.encode(query)
        if query_vec.size == 0:
            return []

        results = self.vector_store.search(query_vec[0], k=k)

        formatted: List[Dict[str, Any]] = []
        for r in results:
            dist = r.get("distance")
            similarity = None
            if isinstance(dist, (int, float)):
                similarity = 1.0 - float(dist)

            formatted.append({
                "metadata": r.get("metadata", {}),
                "similarity": similarity,
                "distance": dist,
            })

        return formatted
