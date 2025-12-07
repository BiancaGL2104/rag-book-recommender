# src/service/book_recommendation_service.py

from typing import Dict, Any

from src.pipeline.rag_pipeline import RAGPipeline
from src.generator.rag_generator import RAGGenerator
from src.retriever.retriever import Retriever


class BookRecommendationService:
    def __init__(self):
        """
        Initialize:
        - Retriever (handles loading VectorStore internally)
        - RAGGenerator (LLM wrapper for Ollama)
        - RAGPipeline (retrieval → rerank → context → LLM)
        """

        # 1) Build retriever.
        #    If your Retriever.__init__ already has default paths, you can just do: Retriever()
        #    Otherwise, pass the correct paths for index + metadata:
        self.retriever = Retriever(
            index_path="models/faiss_index.bin",
            metadata_path="models/metadata.pkl",
        )

        # 2) Build generator (wrapper around generator_ollama.Generator)
        self.generator = RAGGenerator(model="llama3")

        # 3) Build RAG pipeline
        self.pipeline = RAGPipeline(
            retriever=self.retriever,
            generator=self.generator,
        )

    def recommend(self, query: str, style: str | None = None) -> Dict[str, Any]:
        """
        Public API called by the Streamlit UI.
        """
        return self.pipeline.run(query=query, style=style, debug=False)
