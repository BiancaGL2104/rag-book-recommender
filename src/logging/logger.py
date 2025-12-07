import json
import time
from pathlib import Path

def log_result(result, log_path="data/logs/results.jsonl"):
    Path("data/logs").mkdir(parents=True, exist_ok=True)
    
    entry = {
        "timestamp": time.time(),
        "query": result["query"],
        "retrieved_titles": [d["title"] for d in result["retrieved"]],
        "top_scores": [d["final_score"] for d in result["retrieved"][:5]],
        "context": result["context"],
        "llm_answer": result["answer"],
        "recommended_books": result["recommended_books"],
    }

    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
