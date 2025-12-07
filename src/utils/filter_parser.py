# src/utils/filter_parser.py

import re
from typing import Dict, Any


def parse_filters(query: str) -> Dict[str, Any]:
    """
    Parse simple numeric constraints from a natural language query.

    Supports:
      - "under 300 pages", "below 400 pages"
      - "over 500 pages", "more than 200 pages"
      - "above 4 stars", "over 4.5 stars", "rating at least 4 stars"

    Returns a dict like:
      {
        "max_pages": 300,
        "min_pages": 500,
        "min_rating": 4.0
      }
    """
    filters: Dict[str, Any] = {}
    q = query.lower()

    # -----------------------------
    # PAGES: maximum (under / below)
    # -----------------------------
    m = re.search(r"(under|below|less than)\s+(\d+)\s+pages", q)
    if m:
        try:
            filters["max_pages"] = int(m.group(2))
        except ValueError:
            pass

    # Also support: "< 300 pages"
    m = re.search(r"<\s*(\d+)\s*pages", q)
    if m and "max_pages" not in filters:
        try:
            filters["max_pages"] = int(m.group(1))
        except ValueError:
            pass

    # -----------------------------
    # PAGES: minimum (over / more than)
    # -----------------------------
    m = re.search(r"(over|more than|at least)\s+(\d+)\s+pages", q)
    if m:
        try:
            filters["min_pages"] = int(m.group(2))
        except ValueError:
            pass

    # Also support: "> 300 pages"
    m = re.search(r">\s*(\d+)\s*pages", q)
    if m and "min_pages" not in filters:
        try:
            filters["min_pages"] = int(m.group(1))
        except ValueError:
            pass

    # -----------------------------
    # RATING: minimum
    # -----------------------------
    # e.g., "above 4 stars", "over 4.5 stars", "rating at least 4 stars"
    m = re.search(
        r"(above|over|at least)\s+(\d(\.\d)?)\s+stars", q
    )
    if m:
        try:
            filters["min_rating"] = float(m.group(2))
        except ValueError:
            pass

    # Also support: "rating >= 4" or ">= 4 stars"
    m = re.search(r">=\s*(\d(\.\d)?)\s*stars?", q)
    if m and "min_rating" not in filters:
        try:
            filters["min_rating"] = float(m.group(1))
        except ValueError:
            pass

    return filters
