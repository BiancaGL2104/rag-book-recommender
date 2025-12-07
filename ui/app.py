# ui/app.py
import os
import sys
import random

# ---- Make project root importable as a package root ----
this_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(this_file_dir)  # one level up from ui/

if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
from src.service.book_recommendation_service import BookRecommendationService

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="CookMate – Book RAG",
    page_icon="📚",
    layout="wide",
)

# ---------------------------
# Instantiate service once
# ---------------------------
if "service" not in st.session_state:
    st.session_state["service"] = BookRecommendationService()

service = st.session_state["service"]

# ---------------------------
# Initialize chat state
# ---------------------------
if "messages" not in st.session_state:
    # list of {"role": "user"|"assistant", "content": str}
    st.session_state["messages"] = []

# Store last pipeline result for debug panels
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.header("⚙️ Settings")

style = st.sidebar.selectbox(
    "Answer style",
    ["default", "friendly", "formal", "concise", "detailed"],
    index=0,
)

if style == "default":
    style_param = None
else:
    style_param = style

st.sidebar.markdown(
    """
**Style modes:**
- `friendly` – warm, casual tone  
- `formal` – more academic / precise  
- `concise` – short answers  
- `detailed` – more explanation  
"""
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
**You can also use filters in your query:**
- *"fantasy under 300 pages"*
- *"romance above 4 stars"*
- *"mystery over 400 pages with twists"*
"""
)

# Surprise me button
if st.sidebar.button("🎲 Surprise me"):
    surprise_queries = [
        "recommend me a cozy fantasy romance under 400 pages",
        "give me a fast-paced thriller with twists above 4 stars",
        "recommend a lighthearted rom-com with humor",
        "give me a dark fantasy with mature themes",
    ]
    user_query = random.choice(surprise_queries)
    # add fake user turn so it appears in chat
    st.session_state["messages"].append({"role": "user", "content": user_query})

    result = service.recommend(query=user_query, style=style_param)
    answer = result["answer"]

    st.session_state["messages"].append({"role": "assistant", "content": answer})
    st.session_state["last_result"] = result

# Clear chat button
if st.sidebar.button("🔄 Clear chat"):
    st.session_state["messages"] = []
    st.session_state["last_result"] = None

# ---------------------------
# Main title
# ---------------------------
st.title("📚 Book Recommender")
st.write("Ask for a type of book and get tailored recommendations in chat form.")

st.caption(
    """
Examples you can try:
- *"cozy fantasy romance with light vibes"*
- *"dark academia with atmospheric writing"*
- *"YA dystopian like Hunger Games under 400 pages"*
- *"mystery thriller with unexpected plot twists above 4 stars"*
"""
)

# ---------------------------
# Render chat history
# ---------------------------
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---------------------------
# Chat input
# ---------------------------
user_query = st.chat_input("What are you in the mood to read?")

if user_query:
    # 1) Add user message to history
    st.session_state["messages"].append({"role": "user", "content": user_query})

    # 2) Call pipeline
    if not user_query.strip():
        with st.chat_message("assistant"):
            st.warning("Please enter a query.")
    else:
        result = service.recommend(query=user_query, style=style_param)
        answer = result["answer"]

        # 3) Show assistant answer
        with st.chat_message("assistant"):
            st.write(answer)

        # 4) Store assistant message + result
        st.session_state["messages"].append(
            {"role": "assistant", "content": answer}
        )
        st.session_state["last_result"] = result

# ---------------------------
# Debug / retrieval panels (for last response)
# ---------------------------
last_result = st.session_state.get("last_result")

if last_result is not None:
    st.markdown("---")
    st.subheader("📖 Retrieved Books (reranked)")

    for item in last_result["retrieved"][:5]:
        # Item may look like:
        # 1) {"metadata": {...}, "score": ...}
        # 2) {"title": ..., "author": ..., "final_score": ...}

        # Extract metadata (fallback to item itself)
        meta = item.get("metadata", item)

        title = meta.get("title", "Unknown")
        author = meta.get("author", "Unknown")

        # Normalize genres for pretty display
        genres_raw = meta.get("genres", "Unknown")
        if isinstance(genres_raw, list):
            genres = ", ".join(genres_raw)
        else:
            genres = str(genres_raw)

        rating = meta.get("rating", "N/A")

        # Try multiple score keys depending on pipeline output
        score = (
            item.get("score")
            or item.get("final_score")
            or meta.get("score")
            or 0
        )

        st.markdown(
            f"**{title}** by {author}  \n"
            f"Genres: {genres}  \n"
            f"Rating: {rating}  \n"
            f"Score: {score:.3f}"
        )
        st.write("---")

    if "context" in last_result:
        with st.expander("🔍 Show retrieval context sent to the LLM"):
            st.code(last_result["context"], language="markdown")

    # Optional debug block if you store detailed scores
    debug_keys = ["similarity", "score"]
    sample_item = last_result["retrieved"][0]
    if all(k in sample_item for k in debug_keys):
        with st.expander("📊 Debug: reranking scores"):
            for item in last_result["retrieved"][:5]:
                st.write(
                    f"{item['metadata'].get('title', 'Unknown')} — "
                    f"sim={item.get('similarity', 0):.3f}, "
                    f"final={item.get('score', 0):.3f}"
                )
