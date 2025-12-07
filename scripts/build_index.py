import os
import sys

this_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(this_file_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd 
from src.retriever.embedder import Embedder
from src.retriever.vector_store import VectorStore
from src.retriever.load_data import load_books

def build_corpus(df):
    """
    Turn each book row into:
    - text: what we'll embed -> df['retrieval_text']
    - metadata: what we'll return to the user
    """
    texts = []
    metas = []

    for _, row in df.iterrows():
        title = str(row["Title"]) if pd.notna(row["Title"]) else ""
        author = str(row["Author"]) if pd.notna(row["Author"]) else ""
        genres = str(row["genres"]) if pd.notna(row["genres"]) else ""
        publisher = str(row["publisher"]) if pd.notna(row["publisher"]) else None
        year = int(row["year"]) if pd.notna(row["year"]) else None

        rating = float(row["average_rating"]) if pd.notna(row["average_rating"]) else None

        retrieval_text = str(row["retrieval_text"]) if pd.notna(row["retrieval_text"]) else ""

        if retrieval_text.strip() == "":
            retrieval_text = f"{title}. Genres: {genres}. Author: {author}"
        
        text = retrieval_text

        meta = {
            "book_id": int(row["Book Id"]) if pd.notna(row["Book Id"]) else None,
            "title": title,
            "author": author,
            "genres": genres,
            "rating": rating,
            "year": year,
            "publisher": publisher,
        }

        texts.append(text)
        metas.append(meta)

    return texts, metas

def main():
    df = load_books("data/clean_books.csv")
    if df is None or len(df) == 0:
        print("No data to index. Make sure clean_books.csv exists and is not empty.")
        return
    
    print(f"Building corpus from {len(df)} books...")
    texts, metas = build_corpus(df)
    print(f"Prepared {len(texts)} texts for embedding.")

    embedder = Embedder()

    batch_size = 64
    n = len(texts)
    vector_store = None

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_texts = texts[start: end]
        batch_metas = metas[start:end]

        print(f"Embedding batch {start}-{end} / {n}")
        batch_vectors = embedder.encode(batch_texts)

        if vector_store is None:
            dim = batch_vectors.shape[1]
            print(f"Initializing VectorStore with dim = {dim}")
            vector_store = VectorStore(dim=dim)
        
        vector_store.add(batch_vectors, batch_metas)

    index_path = "models/faiss_index.bin"
    meta_path = "models/metadata.pkl"

    vector_store.save(index_path=index_path, meta_path=meta_path)
    print("Index building complete.")

if __name__ == "__main__":
    cwd = os.path.basename(os.getcwd())
    if cwd == "scripts":
        print(
            "Pkease run this script from the project root, e.g.:"
            "'python3 scripts/build_index.py'"
        )
    else:
        main()