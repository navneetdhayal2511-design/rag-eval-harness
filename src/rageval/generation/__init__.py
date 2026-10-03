from typing import Any

from rageval.generation.base import ABSTAIN_TEXT, Answerer, ExtractiveAnswerer

__all__ = ["ABSTAIN_TEXT", "Answerer", "ExtractiveAnswerer", "build_answerer"]


def build_answerer(name: str, **kwargs: Any) -> Answerer:
    if name == ExtractiveAnswerer.name:
        return ExtractiveAnswerer(**kwargs)
    if name == "openai":
        from rageval.generation.llm import OpenAIAnswerer

        return OpenAIAnswerer(**kwargs)
    raise ValueError(f"unknown answerer {name!r}; available: ['extractive', 'openai']")
