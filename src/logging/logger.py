# src/logging/logger.py

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Any, List


def _safe_get_title(item: Dict[str, Any]) -> str:
    """
    Extract title safely from either:
    - {"title": ...}
    - {"metadata": {"title": ...}}
    """
    if "metadata" in item and isinstance(item["metadata"], dict):
        return str(item["metadata"].get("title", ""))
    return str(item.get("title", ""))


def _safe_get_score(item: Dict[str, Any]) -> float | None:
    """
    Extract a score safely from different pipeline outputs.
    """
    for key in ("final_score", "score", "similarity"):
        if key in item:
            try:
                return float(item[key])
            except (TypeError, ValueError):
                return None
    return None


def log_result(result: Dict[str, Any], base_dir: str = "data/logs") -> None:
    """
    Append one RAG interaction to a JSONL log file.

    This function is deliberately defensive:
    - logging failures must NEVER break the app
    - schema differences are handled gracefully

    Parameters
    ----------
    result : dict
        Output from RAGPipeline.run()
    base_dir : str
        Directory where logs are stored
    """
    try:
        log_dir = Path(base_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "results.jsonl"

        retrieved = (
            result.get("retrieved_books")
            or result.get("retrieved")
            or []
        )

        entry = {
            "timestamp": time.time(),
            "query": result.get("query", ""),
            "retrieved_titles": [
                _safe_get_title(d) for d in retrieved
                if _safe_get_title(d)
            ],
            "top_scores": [
                s for s in (
                    _safe_get_score(d) for d in retrieved[:5]
                ) if s is not None
            ],
            "context": result.get("context", ""),
            "llm_answer": result.get("answer", ""),
            "recommended_books": result.get("recommended_books", []),
        }

        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    except Exception:
        return
