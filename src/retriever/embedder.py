# src/retriever/embedder.py

from __future__ import annotations

from typing import List, Union
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class Embedder:
    """
    Sentence embedding wrapper based on SentenceTransformers.

    Responsibilities:
    - load embedding model
    - encode text into dense vectors
    - normalize vectors for similarity search
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str | None = None):
        """
        Parameters
        ----------
        model_name : str
            Name of the SentenceTransformer model.
        device : str, optional
            "cpu", "cuda", or None (auto-detect).
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading embedding model: {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Encode one or more texts into normalized embedding vectors.

        Parameters
        ----------
        texts : str or List[str]
            Input text(s).

        Returns
        -------
        np.ndarray
            Array of shape (n, dim) with L2-normalized embeddings.
        """
        if not texts:
            return np.empty((0, self.model.get_sentence_embedding_dimension()))

        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.clip(norms, a_min=1e-12, a_max=None)

        return embeddings
