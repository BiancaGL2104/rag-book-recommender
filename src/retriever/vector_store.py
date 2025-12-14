# src/retriever/vector_store.py

from __future__ import annotations

from typing import List, Dict, Any, Optional
from pathlib import Path
import pickle
import os

import faiss
import numpy as np


class VectorStore:
    """
    Wrapper around a FAISS index and its associated metadata.

    Responsibilities:
    - store dense vectors in FAISS
    - keep metadata aligned with vector positions
    - provide similarity search
    """

    def __init__(
        self,
        dim: Optional[int] = None,
        index: Optional[faiss.Index] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Initialize a VectorStore.

        Either:
        - pass `dim` to create a new empty FAISS index, or
        - pass an existing FAISS `index`.

        Parameters
        ----------
        dim : int, optional
            Embedding dimension for a new index.
        index : faiss.Index, optional
            Preloaded FAISS index.
        metadata : list of dict, optional
            Metadata entries aligned with index vectors.
        """
        if index is not None:
            self.index = index
        elif dim is not None:
            self.index = faiss.IndexFlatL2(dim)
        else:
            raise ValueError("Must provide either 'dim' or 'index'.")

        self.metadata: List[Dict[str, Any]] = metadata or []

    @property
    def dim(self) -> int:
        return self.index.d

    # ---------------------------
    # Index operations
    # ---------------------------

    def add(self, vectors: np.ndarray, metas: List[Dict[str, Any]]) -> None:
        """
        Add vectors and corresponding metadata to the index.

        Parameters
        ----------
        vectors : np.ndarray
            Array of shape (n, dim).
        metas : list of dict
            Metadata entries of length n.
        """
        vectors = np.asarray(vectors, dtype="float32")

        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(
                f"Vector dimension mismatch: expected (*, {self.dim}), "
                f"got {vectors.shape}"
            )

        if len(vectors) != len(metas):
            raise ValueError(
                "Number of vectors and metadata entries must match."
            )

        self.index.add(vectors)
        self.metadata.extend(metas)

    def search(self, vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the index for the k nearest neighbors of a query vector.

        Parameters
        ----------
        vector : np.ndarray
            Single embedding of shape (dim,).
        k : int
            Number of neighbors to retrieve.

        Returns
        -------
        List[Dict[str, Any]]
            Each item contains:
            {
                "metadata": dict,
                "distance": float
            }
        """
        vector = np.asarray(vector, dtype="float32").reshape(1, -1)

        distances, indices = self.index.search(vector, k)

        results: List[Dict[str, Any]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            results.append({
                "metadata": self.metadata[idx],
                "distance": float(dist),
            })

        return results

    # ---------------------------
    # Metadata access
    # ---------------------------

    def get_all_metadata(self) -> List[Dict[str, Any]]:
        """
        Return a shallow copy of all metadata entries.
        """
        return list(self.metadata)

    # ---------------------------
    # Persistence
    # ---------------------------

    def save(
        self,
        index_path: str | Path = "data/models/faiss_index.bin",
        meta_path: str | Path = "data/models/metadata.pkl",
    ) -> None:
        """
        Persist the FAISS index and metadata to disk.
        """
        index_path = Path(index_path)
        meta_path = Path(meta_path)

        index_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_path))
        with meta_path.open("wb") as f:
            pickle.dump(self.metadata, f)

        print(f"Saved FAISS index to {index_path}")
        print(f"Saved metadata to {meta_path}")

    @classmethod
    def load(
        cls,
        index_path: str | Path = "data/models/faiss_index.bin",
        meta_path: str | Path = "data/models/metadata.pkl",
    ) -> "VectorStore":
        """
        Load a FAISS index and metadata from disk.
        """
        index_path = Path(index_path)
        meta_path = Path(meta_path)

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {meta_path}")

        index = faiss.read_index(str(index_path))
        with meta_path.open("rb") as f:
            metadata = pickle.load(f)

        print(f"Loaded FAISS index from {index_path}")
        print(f"Loaded metadata from {meta_path} (n={len(metadata)})")
        return cls(index=index, metadata=metadata)
