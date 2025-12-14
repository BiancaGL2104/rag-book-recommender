# src/retriever/load_data.py

from __future__ import annotations

from pathlib import Path
from typing import Optional, List

import pandas as pd


REQUIRED_COLUMNS: List[str] = [
    "Book Id",
    "Title",
    "Author",
    "genres",
    "average_rating",
]


def load_books(path: str | Path = "data/clean_books.csv") -> pd.DataFrame:
    """
    Load the cleaned books dataset.

    Parameters
    ----------
    path : str or Path
        Path to the cleaned CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the dataset file does not exist.
    ValueError
        If required columns are missing.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path.resolve()}")

    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}"
        )

    print(f"Loaded dataset with {len(df)} books from {path}")
    return df
