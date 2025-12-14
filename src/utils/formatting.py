# src/utils/formatting.py

"""
Formatting utilities for:
- converting retrieved books into LLM-readable context
- building clean system and user prompts

These functions help standardize how information is passed to the generator.
"""

from typing import List, Dict, Optional

def format_retrieved_books(books: List[Dict]) -> str:
    """
    Convert retrieved book metadata into a structured list for the RAGGenerator.

    Input: list of metadata dicts, each containing (preferred keys):
        - "title"
        - "author"
        - "genres"
        - "rating" or "average_rating"
        - "year"
        - "publisher"
        - "retrieval_text" (optional)

    Output: numbered list of normalized metadata such as:

    1. Book Title by Author | genres: Fantasy, Adventure | rating: 4.32 | year: 2018 | publisher: Tor Books

    This text is fed directly into the generator. No hallucinated fields should appear.
    """

    lines = []

    for i, meta in enumerate(books, start=1):
        b = meta.get("metadata", meta)

        title = b.get("title") or b.get("Title") or "Unknown title"
        author = b.get("author") or b.get("Author") or "Unknown author"

        genres = b.get("genres", None)
        year = b.get("year", None)
        publisher = b.get("publisher") or b.get("Publisher")

        rating = (
            b.get("rating")
            or b.get("average_rating")
            or None
        )

        parts = [f"{i}. {title} by {author}"]

        if genres:
            if isinstance(genres, list):
                genres_str = ", ".join(str(g).strip() for g in genres)
            else:
                genres_str = str(genres)
            parts.append(f"genres: {genres_str}")

        if rating is not None:
            try:
                parts.append(f"rating: {float(rating):.2f}")
            except Exception:
                parts.append(f"rating: {rating}")

        if year is not None:
            parts.append(f"year: {year}")

        if publisher:
            parts.append(f"publisher: {publisher}")

        lines.append(" | ".join(parts))

    return "\n".join(lines)

def build_system_prompt(style: Optional[str] = None) -> str:
    """
    Build a consistent system prompt with style modifiers.

    This is mostly used in old notebooks.
    Modern code uses RAGGenerator's system prompt.
    """
    base = (
        "You are a book recommendation assistant operating on top of a retrieval system. "
        "You must ONLY recommend books from the provided retrieved list. "
        "Do NOT invent authors, titles, plots, or metadata. "
        "If no books match, state this clearly and suggest the closest fits. "
        "Your answers must remain grounded in the retrieved context."
    )

    if style == "funny":
        base += " Use a light and humorous tone while remaining respectful and clear."
    elif style == "formal":
        base += " Use a formal and academically appropriate tone."
    elif style == "short":
        base += " Keep answers extremely concise (1–2 sentences per book)."
    elif style == "detailed":
        base += " Provide detailed explanations (3–4 sentences per book)."

    return base

def build_user_prompt(
    user_query: str,
    books: List[Dict],
    max_suggestions: int = 3,
) -> str:
    """
    Build a clean user prompt for generator experiments in the notebooks.

    Not used in the main application (the RAGGenerator builds its own prompt),
    but kept for testing and reproducibility.
    """
    books_text = format_retrieved_books(books)

    prompt = f"""USER QUERY:
{user_query}

RETRIEVED BOOKS (you MUST recommend only from these):
{books_text}

TASK:
Choose up to {max_suggestions} books that best match the user's request.
For each selected book:
- mention the title and author,
- explain why it fits using genres, topics, or rating,
- do NOT mention books not in the list above.

If none of the books seem relevant, say so clearly and suggest the closest matches anyway.
""".strip()

    return prompt
