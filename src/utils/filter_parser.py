# src/utils/filter_parser.py

"""
Lightweight parser for extracting numeric filters from natural language queries.

Currently supports:
- Page constraints:
    - "under 300 pages", "below 400 pages", "less than 250 pages"
    - "over 500 pages", "more than 200 pages", "at least 350 pages"
    - " < 300 pages", " > 400 pages"
- Rating constraints:
    - "above 4 stars", "over 4.5 stars", "at least 4 stars"
    - "rating above 4.2", "rating at least 3.8"
    - ">= 4 stars"

Returns a dict such as:
    {
        "max_pages": 300,
        "min_pages": 500,
        "min_rating": 4.0
    }

The actual application of these filters is done in the retriever / pipeline.
"""

import re
from typing import Dict, Any


Number = str  


def _parse_int(value: Number) -> Any:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_float(value: Number) -> Any:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_filters(query: str) -> Dict[str, Any]:
    """
    Extract simple numeric constraints from a natural language query.

    Parameters
    ----------
    query : str
        User query (free text).

    Returns
    -------
    Dict[str, Any]
        Dictionary of constraints. Possible keys:
            - "max_pages": int
            - "min_pages": int
            - "min_rating": float
    """
    filters: Dict[str, Any] = {}
    q = (query or "").lower()

    m = re.search(r"(under|below|less than)\s+(\d+)\s+pages", q)
    if m:
        v = _parse_int(m.group(2))
        if v is not None:
            filters["max_pages"] = v

    if "max_pages" not in filters:
        m = re.search(r"<\s*(\d+)\s*pages", q)
        if m:
            v = _parse_int(m.group(1))
            if v is not None:
                filters["max_pages"] = v

    m = re.search(r"(over|more than|at least)\s+(\d+)\s+pages", q)
    if m:
        v = _parse_int(m.group(2))
        if v is not None:
            filters["min_pages"] = v

    if "min_pages" not in filters:
        m = re.search(r">\s*(\d+)\s*pages", q)
        if m:
            v = _parse_int(m.group(1))
            if v is not None:
                filters["min_pages"] = v

    rating_pattern = r"(\d+(?:\.\d+)?)"

    m = re.search(
        rf"(above|over|at least)\s+{rating_pattern}\s+stars?",
        q,
    )
    if m:
        v = _parse_float(m.group(2))
        if v is not None:
            filters["min_rating"] = v

    if "min_rating" not in filters:
        m = re.search(
            rf"rating\s+(above|over|at least)\s+{rating_pattern}",
            q,
        )
        if m:
            v = _parse_float(m.group(2))
            if v is not None:
                filters["min_rating"] = v

    if "min_rating" not in filters:
        m = re.search(
            rf">=\s*{rating_pattern}\s*stars?",
            q,
        )
        if m:
            v = _parse_float(m.group(1))
            if v is not None:
                filters["min_rating"] = v

    return filters
