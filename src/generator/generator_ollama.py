# src/generator/generator_ollama.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any
import time

import ollama


@dataclass
class OllamaConfig:
    """
    Configuration for Ollama generation.
    """
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    timeout_s: Optional[float] = 60.0  
    retries: int = 1
    retry_backoff_s: float = 0.7


class Generator:
    """
    Thin wrapper around Ollama chat completions.

    Responsibilities:
    - build chat messages list
    - call ollama.chat()
    - handle errors + empty responses robustly

    Notes:
    - Some Ollama Python client versions do not implement a true request timeout.
      We still keep timeout_s as a config so you can enforce it externally if needed.
    """

    def __init__(self, model: str = "llama3", config: Optional[OllamaConfig] = None):
        self.model = model
        self.config = config or OllamaConfig()

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate a response from Ollama.

        Parameters
        ----------
        system_prompt : str
            System instructions.
        user_prompt : str
            User query + retrieved context.
        temperature : float, optional
            Overrides config temperature.
        max_tokens : int, optional
            Overrides config max_tokens.

        Returns
        -------
        str
            Assistant message content. In case of failure, returns a friendly
            fallback message (not a Python exception string).
        """
        temp = temperature if temperature is not None else self.config.temperature
        mtok = max_tokens if max_tokens is not None else self.config.max_tokens

        messages = [
            {"role": "system", "content": system_prompt or ""},
            {"role": "user", "content": user_prompt or ""},
        ]

        options: Dict[str, Any] = {"temperature": float(temp)}
        if mtok is not None:
            try:
                options["num_predict"] = int(mtok)
            except (TypeError, ValueError):
                pass

        last_error: Optional[Exception] = None

        for attempt in range(self.config.retries + 1):
            try:
                response = ollama.chat(
                    model=self.model,
                    messages=messages,
                    options=options,
                )

                content = (response.get("message") or {}).get("content") or ""
                content = content.strip()

                if not content:
                    return (
                        "I couldn't generate a response just now. "
                        "Please try rephrasing your request or try again."
                    )

                return content

            except Exception as e:
                last_error = e
                if attempt < self.config.retries:
                    time.sleep(self.config.retry_backoff_s * (attempt + 1))
                    continue

        return (
            "I’m having trouble generating an answer right now. "
            "Please try again in a moment."
        )

    def healthcheck(self) -> bool:
        """
        Minimal health check: attempts a short call to the model.
        Useful for demos or CI scripts.
        """
        try:
            _ = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": "ping"}],
                options={"temperature": 0.0, "num_predict": 5},
            )
            return True
        except Exception:
            return False
