# ui/pages/2_Analytics.py

import os
import sys
import math
from collections import Counter
from typing import List, Dict, Any

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

this_file_dir = os.path.dirname(os.path.abspath(__file__))
ui_dir = os.path.dirname(this_file_dir)   
project_root = os.path.dirname(ui_dir)          
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.service.book_recommendation_service import BookRecommendationService 

if "service" not in st.session_state:
    st.session_state["service"] = BookRecommendationService()

service: BookRecommendationService = st.session_state["service"]


@st.cache_data(show_spinner=False)
def load_catalog_metadata() -> List[Dict[str, Any]]:
    """
    Safely load the catalog metadata from the vector store.
    Uses VectorStore.get_all_metadata() when available.
    """
    try:
        store = service.get_vector_store()
        if store is None:
            return []

        if hasattr(store, "get_all_metadata"):
            return store.get_all_metadata()

        if hasattr(store, "metadata"):
            return list(getattr(store, "metadata"))

        return []
    except Exception:
        return []


metadata: List[Dict[str, Any]] = load_catalog_metadata()

st.title("📊 Analytics Dashboard")
st.write(
    """
Explore statistics from the catalog and your current session:
- **Most recommended books** (session)
- **Genre distribution** (catalog)
- **Rating distribution** (catalog)
"""
)

with st.expander("🔎 Data quality snapshot", expanded=False):
    if not metadata:
        st.info("No catalog metadata loaded.")
    else:
        missing_genres = 0

        for m in metadata:
            genres = m.get("genres")

            if genres is None:
                missing_genres += 1
            elif isinstance(genres, str) and not genres.strip():
                missing_genres += 1
            elif isinstance(genres, list) and len(genres) == 0:
                missing_genres += 1

        missing_ratings = sum(1 for m in metadata if (m.get("rating") is None and m.get("average_rating") is None))
        st.write(f"Catalog size: **{len(metadata)}**")
        st.write(f"Missing genres: **{missing_genres}**")
        st.write(f"Missing ratings: **{missing_ratings}**")

st.markdown("---")

st.subheader("🏆 Most recommended books (this session)")

recommend_counts = st.session_state.get("recommend_counts", {})

if not recommend_counts:
    st.info(
        "No recommendations tracked yet in this session. "
        "Ask for some books in the main app first."
    )
else:
    items = sorted(recommend_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    df_counts = pd.DataFrame(items, columns=["Title", "Times recommended"])
    st.table(df_counts)

st.markdown("---")

st.subheader("🏷️ Genre distribution (catalog)")

genre_counter = Counter()

def _add_genre(g: str) -> None:
    g_clean = str(g).strip()
    if not g_clean:
        return
    g_norm = g_clean.lower()
    genre_counter[g_norm] += 1

for m in metadata:
    genres = m.get("genres", "")

    if isinstance(genres, list):
        for g in genres:
            _add_genre(g)
    else:
        for g in str(genres).split(","):
            _add_genre(g)

if not genre_counter:
    st.info("No genre information available in metadata.")
else:
    top_n = 15
    most_common = genre_counter.most_common(top_n)
    labels = [g for g, _ in most_common]
    values = [c for _, c in most_common]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values)
    ax.set_ylabel("Count")
    ax.set_title(f"Top {top_n} genres in catalog")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    st.pyplot(fig)

st.markdown("---")

st.subheader("⭐ Rating distribution (catalog)")

ratings: List[float] = []
for m in metadata:
    r = m.get("rating") or m.get("average_rating")
    if r is None:
        continue
    try:
        rf = float(r)
        if not math.isnan(rf):
            ratings.append(rf)
    except Exception:
        continue

if not ratings:
    st.info("No ratings found in metadata.")
else:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(ratings, bins=20)
    ax.set_xlabel("Rating")
    ax.set_ylabel("Number of books")
    ax.set_title("Distribution of average ratings")
    st.pyplot(fig)

st.caption(
    f"Catalog size: {len(metadata)} books. "
    "Session-based stats only count queries from this browser window."
)
