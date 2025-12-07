# src/generator/rag_generator.py

from typing import Any, Dict, Optional
from .generator_ollama import Generator as OllamaGenerator


class RAGGenerator:
    """
    Adapter used by the RAGPipeline.

    The pipeline calls:
        generate(query, context, style=None, extra=None)

    Internally we build system + user prompts and call Ollama.
    """

    def __init__(self, model: str = "llama3"):
        self.ollama = OllamaGenerator(model=model)

    def generate(
        self,
        query: str,
        context: str,
        style: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        
        system_prompt = (
            "You are a helpful book recommender. "
            "Use ONLY the books provided in the context. "
            "Do not invent books that are not in the list. "
            "Explain briefly why each recommended book matches."
        )
        if style:
            system_prompt += f" Answer in a {style} style."

        user_prompt = (
            "USER QUERY:\n"
            f"{query}\n\n"
            "RETRIEVED BOOKS:\n"
            f"{context}\n\n"
            "TASK:\n"
            "From the retrieved books, recommend 2–3 that best match the query. "
            "For each, mention the title and a short explanation. "
            "Only mention books that appear in the list above."
        )

        # 3) Call Ollama model
        raw_text = self.ollama.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
        )

        return {
            "answer": raw_text,
            "books": [],            
            "raw_model_output": raw_text,
        }
