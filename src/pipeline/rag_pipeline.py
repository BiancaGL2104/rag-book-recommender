# src/pipeline/rag_pipeline.py

from typing import Optional, List, Dict, Any
import re

from src.utils.formatting import format_retrieved_books
from src.pipeline.adapters import retrieve_books_for_llm
from src.utils.mood import detect_mood


def _map_style_to_generator(style: Optional[str]) -> Dict[str, Optional[str]]:
    """
    Map high-level UI styles to:
    - generator.style ("short", "detailed", or None)
    - personality ("friendly", "academic", etc.)
    """
    if not style or style == "default":
        return {"gen_style": None, "personality": None}

    style = style.lower()

    if style == "friendly":
        return {"gen_style": None, "personality": "friendly"}
    if style == "formal":
        return {"gen_style": None, "personality": "academic"}
    if style == "concise":
        return {"gen_style": "short", "personality": None}
    if style == "detailed":
        return {"gen_style": "detailed", "personality": None}

    return {"gen_style": None, "personality": None}


def _extract_recommended_titles(answer_text: str) -> List[str]:
    """
    Extract book titles from the LLM answer.

    Expected pattern (from your RAGGenerator):
        * **Title** by Author — ...
    But we allow small formatting variations.

    Returns a list of unique titles in order of appearance.
    """
    if not answer_text:
        return []

    pattern = re.compile(r"^[\*\-\•]\s*\*\*(.+?)\*\*", re.MULTILINE)

    titles: List[str] = []
    for match in pattern.finditer(answer_text):
        title = match.group(1).strip()
        if title:
            titles.append(title)

    seen = set()
    deduped: List[str] = []
    for t in titles:
        key = t.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(t)

    return deduped


def _normalize_title(t: str) -> str:
    """
    Normalize titles for matching:
    - lowercase
    - strip whitespace
    - remove surrounding punctuation
    """
    t = (t or "").strip().casefold()
    t = re.sub(r"^[\W_]+|[\W_]+$", "", t)  
    return t


class RAGPipeline:
    """
    High-level RAG pipeline:
    1. Query → Retriever (FAISS)
    2. Raw FAISS results → adapter → book dicts for LLM
    3. Books → formatted context string
    4. RAGGenerator (Ollama) produces the final answer
    """

    def __init__(self, retriever, generator, k: int = 6):
        self.retriever = retriever
        self.generator = generator
        self.k = k

    def run(
        self,
        query: str,
        style: Optional[str] = "default",
        history: Optional[List[Dict[str, str]]] = None,
        personality: Optional[str] = None,
        mood: Optional[str] = None,
        explain: bool = False,
        second_opinion: bool = False,
    ) -> Dict[str, Any]:

        cleaned_query = (query or "").strip()

        if not cleaned_query:
            return {
                "query": query,
                "retrieved": [],
                "retrieved_books": [],
                "recommended_books": [],
                "context": "",
                "answer": "Please describe what kind of books you are looking for 🙂",
                "raw_model_output": "",
                "style": style,
                "mood": None,
            }

        blocked_keywords = ["suicide", "kill myself", "self-harm", "self harm"]
        if any(kw in cleaned_query.lower() for kw in blocked_keywords):
            return {
                "query": query,
                "retrieved": [],
                "retrieved_books": [],
                "recommended_books": [],
                "context": "",
                "answer": (
                    "I'm here to help with book recommendations, but I can't help with this topic. "
                    "If you're struggling, please consider reaching out to a trusted person or a professional."
                ),
                "raw_model_output": "",
                "style": style,
                "mood": None,
            }

        if mood is None:
            mood = detect_mood(cleaned_query)

        raw_results = self.retriever.retrieve(cleaned_query, k=self.k)

        books = retrieve_books_for_llm(raw_results)

        context = format_retrieved_books(books)

        style_mapping = _map_style_to_generator(style)
        gen_style = style_mapping["gen_style"]
        gen_personality = personality or style_mapping["personality"]

        extra = {
            "explain": explain,
            "second_opinion": second_opinion,
        }

        gen_out = self.generator.generate(
            query=cleaned_query,
            context=context,
            style=gen_style,
            history=history,
            personality=gen_personality,
            mood=mood,
            extra=extra,
        )

        answer_text = gen_out.get("answer", "") or ""

        recommended_titles = _extract_recommended_titles(answer_text)
        recommended_books: List[Dict[str, Any]] = []

        lookup: Dict[str, Dict[str, Any]] = {}
        for b in books:
            title = b.get("title") or b.get("Title") or ""
            lookup[_normalize_title(title)] = b

        for t in recommended_titles:
            key = _normalize_title(t)
            if key in lookup:
                recommended_books.append(lookup[key])

        return {
            "query": cleaned_query,
            "retrieved": raw_results,
            "retrieved_books": books,
            "recommended_books": recommended_books,
            "context": context,
            "answer": answer_text,
            "raw_model_output": gen_out.get("raw_model_output", ""),
            "style": style,
            "mood": mood,
        }
