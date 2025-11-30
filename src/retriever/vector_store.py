import faiss
import numpy as np 

class VectorStore:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []
    
    def add(self, vectors, metas):
        vectors = np.array(vectors).astype("float32")
        self.index.add(vectors)
        self.metadata.extend(metas)

    def search(self, vector, k=5):
        vector = np.array([vector]).astype("float32")
        distances, indices = self.index.search(vector, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append({"metadata": self.metadata[idx], "distance": float(dist)})

        return results