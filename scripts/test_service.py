import os
import sys

this_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(this_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.service.book_recommendation_service import BookRecommendationService

def main():
    service = BookRecommendationService(
        index_path="models/faiss_index.bin",
        metadata_path="models/metadata.pkl",
        model_name="all-MiniLM-L6-v2",
        default_k=5,
    )

    query = "romantic fantasy with magic and a strong female lead"

    recs = service.recommend_books(
        query=query,
        n_results=5,
        min_rating=3.8,
        genre_filter="fantasy",
        year_from=1990,
        year_to=2025,
    )

    print(f"Query: {query}")
    print(f"Got {len(recs)} recommendations:\n")
    for i, b in enumerate(recs, start=1):
        print(f"{i}. {b['title']} by {b['author']}")
        print(f"   Rating: {b['rating']}, Year: {b['year']}")
        print(f"   Genres: {b['genres']}")
        print(f"   Score: {b['score']:.3f}")
        print("")


if __name__ == "__main__":
    main()