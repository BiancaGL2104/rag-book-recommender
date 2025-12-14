# scripts/build_index.py

import os
import sys

this_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(this_file_dir)  

if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd

from src.retriever.embedder import Embedder
from src.retriever.vector_store import VectorStore


def build_corpus(df: pd.DataFrame):
    """
    Build a corpus of texts and corresponding metadata dictionaries
    from the cleaned books DataFrame.

    Expected columns in df:
        - Book Id
        - Title
        - Author
        - average_rating
        - genres
        - year
        - publisher (optional)
        - retrieval_text  (pre-formatted text used for embeddings)

    Returns
    -------
    texts : list[str]
        Texts to embed (one per book).
    metas : list[dict]
        Metadata dictionaries aligned with texts.
    """
    if "retrieval_text" not in df.columns:
        raise ValueError(
            "Expected 'retrieval_text' column in clean_books.csv. "
            "Make sure the preprocessing notebook has been run."
        )

    texts = []
    metas = []

    for _, row in df.iterrows():
        text = str(row.get("retrieval_text", "") or "").strip()
        if not text:
            continue

        genres_raw = row.get("genres", "")
        if isinstance(genres_raw, list):
            genres = genres_raw
        else:
            genres = [g.strip() for g in str(genres_raw).split(",") if g.strip()]

        rating = row.get("average_rating", None)
        try:
            rating = float(rating) if rating is not None else None
        except (TypeError, ValueError):
            rating = None

        year = row.get("year", None)
        try:
            year = int(year) if year is not None else None
        except (TypeError, ValueError):
            year = None

        meta = {
            "book_id": row.get("Book Id"),
            "title": row.get("Title") or "",
            "author": row.get("Author") or "",
            "rating": rating,
            "average_rating": rating, 
            "genres": genres,
            "year": year,
            "publisher": row.get("publisher") or row.get("Publisher") or "",
            "retrieval_text": text,
        }

        texts.append(text)
        metas.append(meta)

    return texts, metas


def main():
    csv_path = os.path.join(project_root, "data", "clean_books.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Could not find clean_books.csv at {csv_path}. "
            "Make sure you ran the preprocessing notebook and saved the file."
        )

    print(f"[INFO] Loading cleaned data from: {csv_path}")
    df = pd.read_csv(csv_path)

    print("[INFO] Building corpus (texts + metadata) ...")
    texts, metas = build_corpus(df)
    print(f"[INFO] Corpus size: {len(texts)} books")

    if not texts:
        raise RuntimeError("Corpus is empty. Check your preprocessing / retrieval_text.")

    print("[INFO] Initializing embedder ...")
    embedder = Embedder(model_name="all-MiniLM-L6-v2")

    print("[INFO] Encoding texts into embeddings ...")
    vectors = embedder.encode(texts)
    dim = vectors.shape[1]
    print(f"[INFO] Embedding dimension: {dim}")

    print("[INFO] Building FAISS index ...")
    vector_store = VectorStore(dim=dim)
    vector_store.add(vectors=vectors, metas=metas)

    index_path = os.path.join(project_root, "models", "faiss_index.bin")
    meta_path = os.path.join(project_root, "models", "metadata.pkl")

    print(f"[INFO] Saving index to: {index_path}")
    print(f"[INFO] Saving metadata to: {meta_path}")
    vector_store.save(index_path=index_path, meta_path=meta_path)

    print("[INFO] Index building complete.")


if __name__ == "__main__":
    cwd_name = os.path.basename(os.getcwd())
    if cwd_name == "scripts":
        print(
            "Please run this script from the project root, e.g.:\n"
            "  python3 scripts/build_index.py\n"
        )
    else:
        main()
