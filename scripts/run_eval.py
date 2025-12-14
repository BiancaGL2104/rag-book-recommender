# scripts/run_eval.py

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

this_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(this_file_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.service.book_recommendation_service import BookRecommendationService

def load_eval_queries(path: Path) -> List[Dict[str, Any]]:
    """
    Load evaluation queries from a JSON file.

    Expected format (eval_queries.json):

    [
      {
        "query": "cozy fantasy romance",
        "relevant_books": ["Book A"]
      },
      {
        "query": "dark academia with atmospheric writing",
        "relevant_books": ["Book B"]
      },
      ...
    ]
    """
    if not path.exists():
        raise FileNotFoundError(f"Could not find evaluation file at: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected eval_queries.json to contain a list of objects.")

    return data


def main():
    eval_dir = Path(project_root) / "data" / "eval"
    queries_path = eval_dir / "eval_queries.json"
    outputs_path = eval_dir / "rag_outputs.jsonl"   
    backup_results_path = eval_dir / "results.jsonl"  

    print(f"[INFO] Loading evaluation queries from: {queries_path}")
    records = load_eval_queries(queries_path)
    print(f"[INFO] Loaded {len(records)} evaluation queries")

    eval_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Initializing BookRecommendationService...")
    service = BookRecommendationService()

    with outputs_path.open("w", encoding="utf-8") as out_f, \
         backup_results_path.open("w", encoding="utf-8") as out_backup:

        for idx, rec in enumerate(records, start=1):
            query = rec.get("query", "")
            gold = rec.get("relevant_books") or rec.get("gold_relevant_books") or []

            print(f"\n[#{idx}] Query: {query}")

            try:
                result = service.recommend(
                    query=query,
                    style="detailed",
                    use_mood=True,
                    explain=True,
                    second_opinion=False,
                )
            except Exception as e:
                print(f"[ERROR] Failed on query #{idx}: {e}")
                result = {
                    "error": str(e),
                }

            record = {
                "query": query,
                "gold_relevant_books": gold,
                "output": result,
            }

            line = json.dumps(record, ensure_ascii=False)
            out_f.write(line + "\n")
            out_backup.write(line + "\n")

    print(f"\n[INFO] Saved model outputs to:\n  - {outputs_path}\n  - {backup_results_path}")


if __name__ == "__main__":
    main()
