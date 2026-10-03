"""LLM-backed answering. Optional; the harness runs without it."""

from __future__ import annotations

import os
import re
from typing import Any

from rageval.generation.base import ABSTAIN_TEXT, Answerer
from rageval.types import Answer, Chunk

_CITATION_RE = re.compile(r"\[(\d+)\]")

SYSTEM_PROMPT = (
    "You answer strictly from the numbered sources provided. "
    "Cite every claim with the bracketed source number it came from, like [2]. "
    "If the sources do not contain the answer, reply with exactly: "
    f"{ABSTAIN_TEXT}"
)


class OpenAIAnswerer(Answerer):
    """Grounded answering with citation parsing and disk caching."""

    name = "openai"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 400,
        cache_dir: str | None = ".rageval_cache",
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._cache_dir = cache_dir
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError(
                    "OPENAI_API_KEY is not set; use the default extractive answerer or export a key"
                )
            from openai import OpenAI

            self._client = OpenAI()
        return self._client

    def answer(self, question: str, contexts: list[Chunk]) -> Answer:
        from rageval.harness.cache import DiskCache

        if not contexts:
            return Answer(text=ABSTAIN_TEXT, citations=[], abstained=True)

        sources = "\n\n".join(f"[{i}] {chunk.text}" for i, chunk in enumerate(contexts, start=1))
        user = f"Sources:\n{sources}\n\nQuestion: {question}"
        cache = DiskCache(self._cache_dir) if self._cache_dir else None
        key = f"{self.model}|{self.temperature}|{SYSTEM_PROMPT}|{user}"

        hit = cache.get("answer", self.model, key) if cache else None
        if hit is not None:
            text = str(hit["text"])
        else:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
            )
            text = (response.choices[0].message.content or "").strip()
            if cache:
                cache.put("answer", self.model, key, {"text": text})

        return self._parse(text, contexts)

    @staticmethod
    def _parse(text: str, contexts: list[Chunk]) -> Answer:
        if text.strip() == ABSTAIN_TEXT:
            return Answer(text=text, citations=[], abstained=True)
        citations: list[str] = []
        for match in _CITATION_RE.finditer(text):
            index = int(match.group(1)) - 1
            if 0 <= index < len(contexts):
                chunk_id = contexts[index].chunk_id
                if chunk_id not in citations:
                    citations.append(chunk_id)
        return Answer(text=text, citations=citations, abstained=False)
