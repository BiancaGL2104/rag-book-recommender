# scripts/test_service.py

import os
import sys

this_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(this_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.service.book_recommendation_service import BookRecommendationService


def pretty_print_recs(result: dict, max_books: int = 3):
    """
    Utility to print a compact summary of the top recommended / retrieved books.
    """
    recommended = result.get("recommended_books") or []
    retrieved = result.get("retrieved_books") or []

    if recommended:
        books = recommended[:max_books]
        header = "Top recommended books:"
    else:
        books = retrieved[:max_books]
        header = "Top retrieved books (no explicit recommendation list):"

    print(header)
    if not books:
        print("  (no books found)")
        return

    for i, b in enumerate(books, start=1):
        meta = b.get("metadata", b)
        title = meta.get("title") or meta.get("Title") or "Unknown title"
        author = meta.get("author") or meta.get("Author") or "Unknown author"
        rating = (
            meta.get("rating")
            or meta.get("average_rating")
            or "N/A"
        )
        genres = meta.get("genres", "")
        if isinstance(genres, list):
            genres_str = ", ".join(genres)
        else:
            genres_str = str(genres)

        print(f"  {i}. {title} — {author}")
        print(f"     Rating: {rating} | Genres: {genres_str}")
        print("")


def main():
    print("[INFO] Initializing BookRecommendationService ...")
    service = BookRecommendationService()

    test_queries = [
        "Recommend contemporary fiction exploring identity and self-development.",
        "Find mystery novels above 4.2 rating under 350 pages.",
        "Suggest science-fiction published after 2010 with strong world-building.",
        "I would like something uplifting and optimistic after a stressful week.",
        "Books that explore ethical dilemmas in modern society.",
    ]

    for i, query in enumerate(test_queries, start=1):
        print("=" * 80)
        print(f"[TEST {i}] Query: {query}\n")

        try:
            result = service.recommend(
                query=query,
                style="detailed",
                use_mood=True,
                explain=True,
                second_opinion=False,
            )
        except Exception as e:
            print(f"[ERROR] Service call failed: {e}")
            continue

        answer = result.get("answer", "").strip()

        print("Model answer:")
        print(answer if answer else "(empty answer)")
        print("")
        pretty_print_recs(result, max_books=3)

    print("=" * 80)
    print("[INFO] Test run finished.")


if __name__ == "__main__":
    main()
