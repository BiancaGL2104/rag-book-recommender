class RAGPipeline:
    def __init__(self, embedder, vector_store, generator=None):
        self.embedder = embedder
        self.vector_store = vector_store
        self.generator = generator
    
    def retrieve(self, query, k=5):
        q_vec = self.embedder.encode(query)[0]
        return self.vector_store.search(q_vec, k=k)

    def generate(self, query, retrieved_docs):
        if not self.generator:
            raise ValueError("No generator model attached yet.")
        return self.generator.generate(query, retrieved_docs)
    
    def run(self, query):
        retrieved = self.retrieve(query)
        answer = self.generate(query, retrieved)
        return answer