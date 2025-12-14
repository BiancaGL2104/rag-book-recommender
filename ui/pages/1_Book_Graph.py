# ui/pages/1_Book_Graph.py

import os
import sys
from typing import List, Dict, Any, Tuple

import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt

this_file_dir = os.path.dirname(os.path.abspath(__file__))
ui_dir = os.path.dirname(this_file_dir)          # ui/
project_root = os.path.dirname(ui_dir)          
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.service.book_recommendation_service import BookRecommendationService  

if "service" not in st.session_state:
    st.session_state["service"] = BookRecommendationService()

service = st.session_state["service"]
retriever = service.retriever  

def normalize_title(t: Any) -> str:
    return str(t or "").strip()

def get_all_titles(metadata: List[Dict[str, Any]]) -> List[str]:
    titles = []
    for m in metadata:
        t = normalize_title(m.get("title") or m.get("Title"))
        if t:
            titles.append(t)
    return sorted(set(titles), key=lambda x: x.casefold())

def pick_book_metadata_by_title(metadata: List[Dict[str, Any]], title: str) -> Dict[str, Any] | None:
    target = title.casefold().strip()
    for m in metadata:
        t = normalize_title(m.get("title") or m.get("Title"))
        if t and t.casefold().strip() == target:
            return m
    return None

def build_base_text(meta: Dict[str, Any]) -> str:
    """
    Create a text representation of the selected book to embed.
    We prefer retrieval_text (already curated) then description.
    """
    title = normalize_title(meta.get("title") or meta.get("Title"))
    author = normalize_title(meta.get("author") or meta.get("Author"))
    genres = meta.get("genres", "")
    if isinstance(genres, list):
        genres_str = ", ".join([str(g).strip() for g in genres if str(g).strip()])
    else:
        genres_str = str(genres).strip()

    body = meta.get("retrieval_text") or meta.get("description") or ""
    body = str(body).strip()

    if len(body) > 600:
        body = body[:600]

    parts = []
    if title:
        parts.append(f"title: {title}")
    if author:
        parts.append(f"author: {author}")
    if genres_str:
        parts.append(f"genres: {genres_str}")
    if body:
        parts.append(body)

    return ". ".join(parts)

st.title("📚 Book Similarity Graph")
st.write(
    """
Select a book and explore the most similar titles in the embedding space  
(using the same embeddings + FAISS index as the recommender).
"""
)

vector_store = getattr(retriever, "vector_store", None)
if vector_store is None:
    st.error("Retriever has no vector_store. Cannot build similarity graph.")
    st.stop()

metadata = vector_store.get_all_metadata()
all_titles = get_all_titles(metadata)

if not all_titles:
    st.warning("No titles found in metadata. Check that FAISS index + metadata are loaded.")
    st.stop()

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    selected_title = st.selectbox("Base book", options=all_titles)

with col2:
    k_neighbors = st.slider("Neighbors", min_value=3, max_value=15, value=8, step=1)

with col3:
    show_distances = st.checkbox("Show distances", value=False)

if not selected_title:
    st.info("Select a book from the dropdown to see the similarity graph.")
    st.stop()

base_meta = pick_book_metadata_by_title(metadata, selected_title)
if base_meta is None:
    st.error("Could not locate selected title in metadata. Try another title.")
    st.stop()

base_text = build_base_text(base_meta)

try:
    q_vec = retriever.embedder.encode(base_text)[0]
except Exception as e:
    st.error(f"Embedding failed: {e}")
    st.stop()

raw = vector_store.search(q_vec, k=k_neighbors + 1)

neighbors: List[Tuple[str, Dict[str, Any], float]] = []
for item in raw:
    m = item.get("metadata", {})
    t = normalize_title(m.get("title") or m.get("Title") or "Unknown title")
    if not t:
        continue
    if t.casefold().strip() == selected_title.casefold().strip():
        continue

    dist = float(item.get("distance", 0.0))
    neighbors.append((t, m, dist))

neighbors = neighbors[:k_neighbors]

if not neighbors:
    st.warning("No neighbors found for this book. Try another title.")
    st.stop()

G = nx.Graph()
G.add_node(selected_title)

for nb_title, nb_meta, dist in neighbors:
    sim = 1.0 / (1.0 + dist)
    G.add_node(nb_title)
    G.add_edge(selected_title, nb_title, weight=sim)

st.subheader("📈 Similarity graph")

fig, ax = plt.subplots(figsize=(7, 5))
pos = nx.spring_layout(G, seed=42)

nx.draw_networkx_nodes(G, pos, node_size=900, ax=ax)
nx.draw_networkx_edges(G, pos, width=1.6, ax=ax)
nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

ax.axis("off")
st.pyplot(fig)

st.subheader("📚 Nearest neighbors")

for nb_title, nb_meta, dist in neighbors:
    author = nb_meta.get("author") or nb_meta.get("Author") or "Unknown author"
    rating = nb_meta.get("rating") or nb_meta.get("average_rating") or "N/A"

    genres_raw = nb_meta.get("genres", "")
    if isinstance(genres_raw, list):
        genres = ", ".join(genres_raw)
    else:
        genres = str(genres_raw)

    if show_distances:
        st.markdown(
            f"**{nb_title}** by {author}  \n"
            f"⭐ {rating}  \n"
            f"Genres: {genres}  \n"
            f"Distance: {dist:.3f}"
        )
    else:
        st.markdown(
            f"**{nb_title}** by {author}  \n"
            f"⭐ {rating}  \n"
            f"Genres: {genres}"
        )
    st.write("---")
