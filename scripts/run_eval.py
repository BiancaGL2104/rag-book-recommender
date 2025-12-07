# scripts/run_eval.py

import os
import sys
import json
from pathlib import Path

# =========================
# Make project root importable
# =========================

this_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(this_file_dir)  # one level up from scripts/

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now imports from src/ will work
from src.service.book_recommendation_service import BookRecommendationService


def main():
    # -------------------------
    # 1) Ensure eval directory exists
    # -------------------------
    eval_dir = Path("data/eval")
    eval_dir.mkdir(parents=True, exist_ok=True)

    eval_queries_path = eval_dir / "eval_queries.json"
    results_path = eval_dir / "results.jsonl"

    # -------------------------
    # 2) Load evaluation queries
    # -------------------------
    if not eval_queries_path.exists():
        # Minimal placeholder so it doesn't crash
        print(f"[WARN] {eval_queries_path} not found. Creating a small example file.")
        example = [
            {
                "query": "cozy fantasy romance",
                "relevant_books": ["Book A"]
            },
            {
                "query": "dark academia with atmospheric writing",
                "relevant_books": ["Book B"]
            }
        ]
        with open(eval_queries_path, "w") as f:
            json.dump(example, f, indent=2)

    with open(eval_queries_path) as f:
        eval_queries = json.load(f)

    print(f"[INFO] Loaded {len(eval_queries)} eval queries from {eval_queries_path}")

    # -------------------------
    # 3) Initialize service (pipeline + retriever + generator)
    # -------------------------
    service = BookRecommendationService()

    # -------------------------
    # 4) Run evaluation
    # -------------------------
    num = 0
    with open(results_path, "w") as out_f:
        for item in eval_queries:
            query = item["query"]
            gold = item.get("relevant_books", [])
            num += 1

            print(f"\n[{num}/{len(eval_queries)}] Query: {query}")

            output = service.recommend(query=query, style=None)

            record = {
                "query": query,
                "gold_relevant_books": gold,
                "output": output,
            }

            out_f.write(json.dumps(record) + "\n")

    print(f"\n[INFO] Saved results to {results_path}")


if __name__ == "__main__":
    main()
