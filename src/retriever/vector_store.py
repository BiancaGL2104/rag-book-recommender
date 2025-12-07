# src/retriever/vector_store.py

import faiss
import numpy as np
import pickle
import os


class VectorStore:
    """
    Wrapper around a FAISS index + metadata.
    """

    def __init__(self, dim=None, index=None, metadata=None):
        """
        Either pass dim to create a new empty index
        or pass an existing FAISS index.
        """
        if index is not None:
            self.index = index
        elif dim is not None:
            self.index = faiss.IndexFlatL2(dim)
        else:
            raise ValueError("Must provide 'dim' or 'index'.")

        self.metadata = metadata or []

    @property
    def dim(self):
        return self.index.d

    def add(self, vectors, metas):
        """
        vectors: np.array [n, dim]
        metas: list of dicts, length n
        """
        vectors = np.array(vectors).astype("float32")
        self.index.add(vectors)
        self.metadata.extend(metas)

    def search(self, vector, k=5):
        """
        vector: single embedding [dim]
        returns: list of {metadata, distance}
        """
        vector = np.array([vector]).astype("float32")
        distances, indices = self.index.search(vector, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append(
                {
                    "metadata": self.metadata[idx],
                    "distance": float(dist),
                }
            )

        return results

    def save(
        self,
        index_path: str = "data/models/faiss_index.bin",
        meta_path: str = "data/models/metadata.pkl",
    ):
        """
        Persist FAISS index + metadata to disk.
        """
        os.makedirs(os.path.dirname(index_path), exist_ok=True)

        faiss.write_index(self.index, index_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

        print(f"Saved index to {index_path}")
        print(f"Saved metadata to {meta_path}")

    @classmethod
    def load(
        cls,
        index_path: str = "data/models/faiss_index.bin",
        meta_path: str = "data/models/metadata.pkl",
    ) -> "VectorStore":
        """
        Load FAISS index + metadata from disk.
        """
        index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)

        print(f"Loaded index from {index_path}")
        print(f"Loaded metadata from {meta_path} (n={len(metadata)})")
        return cls(index=index, metadata=metadata)
