# src/generator/generator_ollama.py

import ollama
from typing import List, Dict, Optional


class Generator:
    """
    Simple wrapper around Ollama's chat API.

    You pass:
      - system_prompt: instructions for the assistant
      - user_prompt: the actual query + retrieved books context
    """

    def __init__(self, model: str = "llama3"):
        self.model = model

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": temperature,
                },
            )
        except Exception as e:
            return f"Sorry, I couldn't generate a response due to an internal error: {e}"

        content = response.get("message", {}).get("content", "")
        if not content:
            return "Sorry, I couldn't generate a response."
        return content