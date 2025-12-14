# ui/app.py

from __future__ import annotations

import os
import sys
import random
import re
import html
from typing import Any, Dict, List, Optional

THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_FILE_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st

from src.service.book_recommendation_service import BookRecommendationService

st.set_page_config(
    page_title="Book RAG",
    page_icon="📚",
    layout="wide",
)

if "service" not in st.session_state:
    st.session_state["service"] = BookRecommendationService()
service: BookRecommendationService = st.session_state["service"]

st.session_state.setdefault("messages", [])
st.session_state.setdefault("last_result", None)
st.session_state.setdefault("recommend_counts", {})

st.sidebar.header("⚙️ Settings")

style = st.sidebar.selectbox(
    "Answer style",
    ["default", "friendly", "formal", "concise", "detailed"],
    index=0,
)
style_param = None if style == "default" else style

use_mood = st.sidebar.checkbox("Use mood detection", value=True)
explain = st.sidebar.checkbox("Explain recommendations", value=False)
second_opinion = st.sidebar.checkbox("Alternative recommendations", value=False)
show_debug = st.sidebar.checkbox("Show debug panels", value=False)

st.sidebar.markdown(
    """
**Style modes:**
- `friendly` – warm, accessible tone  
- `formal` – structured and precise  
- `concise` – short answers  
- `detailed` – more explanation  
"""
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
**You can include constraints in your query:**
- *"mystery rated above 4.2 under 350 pages"*
- *"science fiction after 2010 with strong world-building"*
- *"historical fiction with complex character development"*
"""
)

def _strip_html(text: str) -> str:
    """Remove HTML + common HTML artifacts such as '</div>' that may appear as text (escaped/spaced)."""
    if text is None:
        return ""

    s = str(text)

    s = html.unescape(s)

    s = re.sub(r"</?\s*div\s*>", " ", s, flags=re.IGNORECASE)

    s = re.sub(r"<[^>]*>", " ", s)

    s = re.sub(r"(?i)\b/?div\b", " ", s)

    s = re.sub(r"\s+", " ", s).strip()

    return s




def _extract_meta(book: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize the expected metadata container."""
    if not isinstance(book, dict):
        return {}
    return book.get("metadata", book)


def _get_title(meta: Dict[str, Any]) -> str:
    return str(meta.get("title") or meta.get("Title") or "Unknown title").strip()


def _get_author(meta: Dict[str, Any]) -> str:
    return str(meta.get("author") or meta.get("Author") or "Unknown author").strip()


def _get_rating(meta: Dict[str, Any]) -> str:
    val = meta.get("rating", meta.get("average_rating", "N/A"))
    try:
        if val == "N/A":
            return "N/A"
        return f"{float(val):.2f}".rstrip("0").rstrip(".")
    except Exception:
        return str(val)


def _get_genres(meta: Dict[str, Any]) -> str:
    genres = meta.get("genres", "N/A")
    if isinstance(genres, list):
        return ", ".join([str(g).strip() for g in genres if str(g).strip()])
    return str(genres)


def render_book_card(book: Dict[str, Any]) -> None:
    """
    Render a book card using Streamlit-native components.
    This avoids HTML artifacts like '</div>' entirely.
    """
    meta = _extract_meta(book)

    title = _get_title(meta)
    author = _get_author(meta)
    rating = _get_rating(meta)
    genres_str = _get_genres(meta)

    raw_desc = (
        meta.get("description")
        or meta.get("retrieval_text")
        or meta.get("context")
        or ""
    )

    description = _strip_html(raw_desc)

    if re.search(r"(?i)</?\s*div\s*>|\b/?div\b", description):
        description = ""

    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.caption(f"by {author}")

        col1, col2 = st.columns([1, 4])
        with col1:
            st.write(f"⭐ {rating}")
        with col2:
            st.write(genres_str)

        if description:
            st.write(description[:280] + ("..." if len(description) > 280 else ""))



def update_recommend_counts(books: List[Dict[str, Any]]) -> None:
    counts: Dict[str, int] = st.session_state.get("recommend_counts", {})
    for item in books:
        meta = _extract_meta(item)
        title = (meta.get("title") or meta.get("Title") or "").strip()
        if not title:
            continue
        counts[title] = counts.get(title, 0) + 1
    st.session_state["recommend_counts"] = counts


def run_query(user_query: str) -> Optional[Dict[str, Any]]:
    """Single entry point for calling the service with current UI parameters."""
    with st.spinner("Searching the catalog and composing suggestions..."):
        return service.recommend(
            query=user_query,
            style=style_param,
            use_mood=use_mood,
            explain=explain,
            second_opinion=second_opinion,
        )

if st.sidebar.button("🎲 Surprise me"):
    surprise_queries = [
        "Recommend a reflective novel exploring personal growth and identity.",
        "Find mystery novels rated above 4.2 under 350 pages.",
        "Suggest science fiction published after 2010 with strong world-building.",
        "Recommend historical fiction featuring complex character development.",
    ]
    user_query = random.choice(surprise_queries)
    st.session_state["messages"].append({"role": "user", "content": user_query})

    try:
        result = run_query(user_query)
    except Exception:
        with st.chat_message("assistant"):
            st.error("Sorry, something went wrong while generating recommendations.")
    else:
        answer = (result or {}).get("answer", "")
        st.session_state["messages"].append({"role": "assistant", "content": answer})
        st.session_state["last_result"] = result

        if hasattr(service, "get_recommendation_stats"):
            st.session_state["recommend_counts"] = service.get_recommendation_stats()

if st.sidebar.button("🔄 Clear chat"):
    st.session_state["messages"] = []
    st.session_state["last_result"] = None

st.title("📚 Book Recommender (RAG)")
st.write("Describe what you want to read and the system will retrieve relevant books and generate recommendations.")

st.caption(
    """
Examples:
- *"Recommend contemporary fiction exploring themes of identity and self-development."*
- *"Find mystery novels rated above 4.2 that are under 350 pages."*
- *"Suggest historical novels with strong character arcs and accurate cultural context."*
- *"Recommend an atmospheric, reflective book suitable for slow reading."*
"""
)

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_query = st.chat_input("What would you like to read?")

if user_query is not None:
    cleaned = user_query.strip()
    empty_patterns = {"", '"', "''", '""'}

    if cleaned in empty_patterns:
        with st.chat_message("assistant"):
            st.warning("Please enter a query.")
    else:
        st.session_state["messages"].append({"role": "user", "content": user_query})

        try:
            result = run_query(user_query)
        except Exception:
            with st.chat_message("assistant"):
                st.error(
                    "Sorry, something unexpected happened while generating recommendations. "
                    "Please try again."
                )
        else:
            answer = (result or {}).get("answer", "")

            with st.chat_message("assistant"):
                st.write(answer)

                recommended_books = (result or {}).get("recommended_books") or []
                retrieved_books = (result or {}).get("retrieved_books") or []

                if recommended_books:
                    st.markdown("### 📚 Recommended books (from retrieved candidates)")
                    for b in recommended_books:
                        render_book_card(b)
                    update_recommend_counts(recommended_books)
                elif retrieved_books:
                    st.markdown("### 📚 Top retrieved books")
                    top_shown = retrieved_books[:3]
                    for b in top_shown:
                        render_book_card(b)
                    update_recommend_counts(top_shown)

            st.session_state["messages"].append({"role": "assistant", "content": answer})
            st.session_state["last_result"] = result

            if hasattr(service, "get_recommendation_stats"):
                st.session_state["recommend_counts"] = service.get_recommendation_stats()

last_result = st.session_state.get("last_result")

if show_debug and last_result is not None:
    st.markdown("---")
    st.subheader("📚 Top retrieved books (debug)")

    retrieved_list = last_result.get("retrieved_books") or last_result.get("retrieved") or []

    if not retrieved_list:
        st.info("No retrieved books for the last query.")
    else:
        for item in retrieved_list[:5]:
            render_book_card(item)

    st.subheader("📖 Retrieved Books (debug view)")
    for item in retrieved_list[:5]:
        meta = _extract_meta(item)

        title = _get_title(meta)
        author = _get_author(meta)
        rating = _get_rating(meta)

        genres_raw = meta.get("genres", "")
        genres = ", ".join(genres_raw) if isinstance(genres_raw, list) else str(genres_raw)

        score = item.get("score") or meta.get("score") or 0

        st.markdown(
            f"**{title}** by {author}  \n"
            f"Genres: {genres}  \n"
            f"Rating: {rating}  \n"
            f"Score: {float(score):.3f}"
        )
        st.write("---")

    if "context" in last_result:
        with st.expander("🔍 Show retrieval context sent to the LLM"):
            st.code(last_result["context"], language="markdown")
