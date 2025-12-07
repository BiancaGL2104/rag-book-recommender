# src/pipeline/rag_pipeline.py

from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class RetrievedDoc:
    """
    Normalized view of a retrieved book, used inside the RAG pipeline.
    """
    book_id: str
    title: str
    author: str
    genres: List[str]
    rating: float
    similarity: float
    retrieval_text: str


def format_docs_for_llm(
    docs: List[RetrievedDoc],
    max_docs: int = 5,
    max_text_chars: int = 400,
) -> str:
    """
    Build the context string we send to the LLM.
    """
    lines = []
    for i, doc in enumerate(docs[:max_docs], start=1):
        snippet = (doc.retrieval_text or "").strip()
        if len(snippet) > max_text_chars:
            snippet = snippet[: max_text_chars].rsplit(" ", 1)[0] + "..."

        genre_str = ", ".join(doc.genres) if doc.genres else "Unknown"

        lines.append(
            f"[BOOK {i}]\n"
            f"Title: {doc.title}\n"
            f"Author: {doc.author}\n"
            f"Genres: {genre_str}\n"
            f"Rating: {doc.rating:.2f}\n"
            f"Description: {snippet}\n"
        )

    return "\n".join(lines)


class RAGPipeline:
    def __init__(
        self,
        retriever,
        generator,
        rating_weight: float = 0.2,
        genre_weight: float = 0.2,
        similarity_weight: float = 0.6,
    ):
        self.retriever = retriever
        self.generator = generator
        self.rating_weight = rating_weight
        self.genre_weight = genre_weight
        self.similarity_weight = similarity_weight

    # ----------------- helpers for reranking -----------------

    def _normalize_rating(self, rating: float, max_rating: float = 5.0) -> float:
        return max(0.0, min(rating / max_rating, 1.0))

    def _estimate_genre_overlap(self, query: str, doc: RetrievedDoc) -> float:
        """
        Very simple genre overlap: intersect query tokens with genre tokens.
        """
        query_lower = query.lower()
        q_tokens = set(query_lower.split())

        if not doc.genres:
            return 0.0

        genre_tokens = set(g.lower() for g in doc.genres)
        overlap = len(q_tokens & genre_tokens)

        if overlap == 0:
            return 0.0
        return min(1.0, overlap / (len(q_tokens) + 1))

    def rerank_docs(
        self,
        query: str,
        docs: List[RetrievedDoc],
        debug: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Compute a final_score per doc using similarity + rating + genre_overlap.
        """
        reranked = []
        for doc in docs:
            sim = getattr(doc, "similarity", 0.0) or 0.0
            rating_norm = self._normalize_rating(doc.rating)
            genre_overlap = self._estimate_genre_overlap(query, doc)

            final_score = (
                self.similarity_weight * sim
                + self.rating_weight * rating_norm
                + self.genre_weight * genre_overlap
            )

            reranked.append(
                {
                    "doc": doc,
                    "scores": {
                        "similarity": sim,
                        "rating_norm": rating_norm,
                        "genre_overlap": genre_overlap,
                        "final_score": final_score,
                    },
                }
            )

        reranked.sort(key=lambda x: x["scores"]["final_score"], reverse=True)

        if debug:
            print("\n[DEBUG] Reranking:")
            for item in reranked:
                d = item["doc"]
                s = item["scores"]
                print(
                    f"- {d.title} | sim={s['similarity']:.3f}, "
                    f"rating={s['rating_norm']:.3f}, "
                    f"genre={s['genre_overlap']:.3f}, "
                    f"final={s['final_score']:.3f}"
                )

        return reranked

    # ----------------- main entry point -----------------

    def run(
        self,
        query: str,
        top_k: int = 10,
        n_recs: int = 3,
        style: Optional[str] = None,
        debug: bool = False,
    ) -> Dict[str, Any]:
        """
        1) Retrieve docs with retriever
        2) Normalize into RetrievedDoc objects
        3) Rerank
        4) Build LLM context
        5) Call generator
        6) Return structured result
        """

        # 1) Retrieve using your retriever API: retrieve(query, k=top_k)
        raw_docs = self.retriever.retrieve(query, k=top_k)

        # If nothing found, let generator handle "no results" case
        if not raw_docs:
            gen_output = self.generator.generate(
                query=query,
                context="",
                style=style,
                extra={"retrieved_books": []},
            )
            result = {
                "query": query,
                "retrieved": [],
                "context": "",
                "answer": gen_output.get("answer", ""),
                "recommended_books": gen_output.get("books", []),
                "meta": {"reason": "no_retrieved_docs"},
            }
            return result

        # 2) Convert raw docs (from Retriever) → RetrievedDoc
        docs: List[RetrievedDoc] = []
        for d in raw_docs:
            meta = d["metadata"]

            # genres: stored as string "Fantasy, Romance" → list
            genres_str = meta.get("genres", "")
            genres_list = [g.strip() for g in genres_str.split(",") if g.strip()]

            docs.append(
                RetrievedDoc(
                    book_id=meta.get("book_id", "") or meta.get("id", ""),
                    title=meta.get("title", "Unknown Title"),
                    author=meta.get("author", "Unknown Author"),
                    genres=genres_list,
                    rating=meta.get("rating", 0.0) or 0.0,
                    similarity=d.get("similarity", 0.0),
                    retrieval_text=meta.get("retrieval_text", "") or meta.get("description", ""),
                )
            )

        # 3) Rerank
        reranked = self.rerank_docs(query=query, docs=docs, debug=debug)

        # 4) Build LLM context from top n_recs docs
        top_docs = [item["doc"] for item in reranked[:n_recs]]
        context = format_docs_for_llm(top_docs)

        # 5) Call generator
        gen_output = self.generator.generate(
            query=query,
            context=context,
            style=style,
            extra={
                "retrieved_books": [
                    {
                        "book_id": d.book_id,
                        "title": d.title,
                        "author": d.author,
                        "genres": d.genres,
                        "rating": d.rating,
                    }
                    for d in top_docs
                ]
            },
        )

        # 6) Build final result dict
        result = {
            "query": query,
            "retrieved": [
                {
                    "book_id": item["doc"].book_id,
                    "title": item["doc"].title,
                    "author": item["doc"].author,
                    "genres": item["doc"].genres,
                    "rating": item["doc"].rating,
                    "similarity": item["scores"]["similarity"],
                    "genre_overlap": item["scores"]["genre_overlap"],
                    "final_score": item["scores"]["final_score"],
                }
                for item in reranked
            ],
            "context": context,
            "answer": gen_output.get("answer", ""),
            "recommended_books": gen_output.get("books", []),
            "meta": {
                "num_retrieved": len(raw_docs),
                "num_recommended": len(gen_output.get("books", [])),
            },
        }

        return result
