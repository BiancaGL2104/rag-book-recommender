# src/generator/rag_generator.py

from typing import Any, Dict, Optional, List
from .generator_ollama import Generator as OllamaGenerator


def _format_history(history: List[Dict[str, str]], max_turns: int = 5) -> str:
    """
    Turn chat history into a short text block.
    history = [
        {"user": "...", "bot": "..."},
        ...
    ]
    We only keep the last `max_turns` exchanges.
    """
    if not history:
        return ""

    turns = history[-max_turns:]
    lines = ["Previous conversation:"]
    for turn in turns:
        user_msg = turn.get("user", "").strip()
        bot_msg = turn.get("bot", "").strip()
        if user_msg:
            lines.append(f"User: {user_msg}")
        if bot_msg:
            lines.append(f"Assistant: {bot_msg}")
    return "\n".join(lines)


PERSONALITY_PROMPTS = {
    "friendly": "You are warm, enthusiastic, and encouraging.",
    "academic": "You are precise, objective, and structured, like an academic lecturer.",
    "poetic": "You use light poetic language and gentle metaphors, but stay clear and readable.",
    "sarcastic": "You are mildly sarcastic but never rude or offensive.",
}

class LLMTimeoutError(Exception):
    """Raised when the underlying LLM call times out or fails hard."""


class LLMResponseFormatError(Exception):
    """Raised when the LLM returns an unexpected / unusable response."""


class RAGGenerator:
    """
    Adapter used by the RAGPipeline.

    The pipeline calls:
        generate(
            query, context,
            style=None,
            history=None,
            personality=None,
            mood=None,
            extra=None,
        )

    Internally we build system + user prompts and call Ollama.
    """

    def __init__(self, model: str = "llama3"):
        self.ollama = OllamaGenerator(model=model)

    def generate(
        self,
        query: str,
        context: str,
        style: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        personality: Optional[str] = None,
        mood: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Build robust prompts for the generator and call Ollama.

        Parameters
        ----------
        query : str
            User query / preference description.
        context : str
            Formatted retrieval context (books with metadata) from the pipeline.
        style : Optional[str]
            Optional style hint ("concise", "detailed", etc.).
        history : Optional[List[Dict[str, str]]]
            Previous turns [{"user": "...", "bot": "..."}, ...].
        personality : Optional[str]
            One of "friendly", "academic", "poetic", "sarcastic", or custom.
        mood : Optional[str]
            Simple mood label: "happy", "sad", "neutral", ... (from mood detector).
        extra : Optional[Dict[str, Any]]
            Extra flags like {"explain": True, "second_opinion": True}.
        temperature : float
            Sampling temperature for the LLM.

        Returns
        -------
        Dict[str, Any]
            {
                "answer": str,            # cleaned text to show in UI
                "books": list,            # (kept empty; pipeline may infer later)
                "raw_model_output": str,  # raw LLM response for debugging
            }
        """
        extra = extra or {}
        history = history or []

        system_parts = [
            "You are a book recommendation assistant that works on top of a retrieval system.",
            "You must ONLY recommend books that appear in the provided context list.",
            "Never invent new book titles, authors, years, or genres.",
            "If no book clearly matches, say so honestly and suggest the closest matches.",
            "Always answer in clear bullet points, with **bold** book titles.",
        ]

        if style in {"short", "concise"}:
            system_parts.append("Keep explanations very short (1–2 sentences per book).")
        elif style == "detailed":
            system_parts.append(
                "Give more detailed explanations (3–4 sentences per book) "
                "and highlight specific themes, tropes, and tone."
            )

        if personality:
            desc = PERSONALITY_PROMPTS.get(personality)
            if desc:
                system_parts.append(desc)
            else:
                system_parts.append(f"Adopt a {personality} tone in your writing.")

        if mood:
            system_parts.append(
                f"The user is currently in a '{mood}' mood. "
                "Choose books and wording that are appropriate for that mood."
            )

        if extra.get("explain"):
            system_parts.append(
                "Explicitly mention why each book matches the query "
                "(genres, themes, pacing, rating, tropes, and overall vibe). "
                "Focus on faithfulness to the provided context and avoid hallucinating details."
            )

        if extra.get("second_opinion"):
            system_parts.append(
                "After giving your main recommendations, also provide a clearly "
                "different 'second opinion' section: either alternative books or a "
                "different perspective on the same books. Avoid repeating the exact "
                "same list in the same order unless there is no reasonable alternative."
            )
            temperature = max(float(temperature), 0.9)

        system_prompt = " ".join(system_parts)

        history_text = _format_history(history) if history else ""
        history_block = f"{history_text}\n\n" if history_text else ""

        user_prompt = (
            f"{history_block}"
            f"USER QUERY:\n{query}\n\n"
            "RETRIEVED BOOKS (only recommend from this list):\n"
            f"{context}\n\n"
            "TASK:\n"
            "- Recommend 2–3 books from the list above that best match the query.\n"
            "- For each book, output a bullet point starting with "
            "`* **Title** by Author — ...`.\n"
            "- Do NOT mention books that are not in the retrieved list.\n"
            "- If nothing fits, say that clearly and suggest the closest matches "
            "from the list.\n"
        )

        try:
            raw_text = self.ollama.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
            )
        except Exception as e:
            raise LLMTimeoutError(f"LLM call failed: {e}") from e

        if not isinstance(raw_text, str) or not raw_text.strip():
            raise LLMResponseFormatError("LLM returned an empty or non-string response.")

        answer_clean = raw_text.strip()

        return {
            "answer": answer_clean,
            "books": [],
            "raw_model_output": raw_text,
        }
