from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

import httpx

from prophet_arena_agent.models import ProphetEvent


@dataclass(frozen=True)
class RetrievedSource:
    title: str | None
    url: str
    snippet: str
    published_at: str | None = None
    score: float | None = None


def _clean_query(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:240]


def build_round_queries(event: ProphetEvent, round_index: int) -> list[str]:
    title = event.title
    category = event.category or ""
    rules = event.rules or ""
    outcomes = event.outcomes[:4]

    if round_index == 0:
        queries = [
            f"{title} {category}".strip(),
            f"{title} resolution criteria official source {rules}".strip(),
        ]
    else:
        outcome_text = " ".join(outcomes)
        queries = [
            f"{title} latest official data",
            f"{title} {outcome_text} forecast odds news",
            f"{title} {category} recent developments".strip(),
        ]

    deduped: list[str] = []
    seen: set[str] = set()
    for query in queries:
        cleaned = _clean_query(query)
        if cleaned and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            deduped.append(cleaned)
    return deduped


class ExaRetriever:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.exa.ai",
        rounds: int = 2,
        results_per_query: int = 4,
        max_sources: int = 10,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.rounds = max(0, rounds)
        self.results_per_query = max(1, results_per_query)
        self.max_sources = max(1, max_sources)

    @classmethod
    def from_env(cls) -> "ExaRetriever | None":
        api_key = os.environ.get("EXA_API_KEY")
        if not api_key:
            return None
        return cls(
            api_key=api_key,
            rounds=int(os.environ.get("EXA_SEARCH_ROUNDS", "2")),
            results_per_query=int(os.environ.get("EXA_RESULTS_PER_QUERY", "4")),
            max_sources=int(os.environ.get("EXA_MAX_SOURCES", "10")),
        )

    async def _search_one(self, query: str) -> list[RetrievedSource]:
        payload: dict[str, Any] = {
            "query": query,
            "numResults": self.results_per_query,
            "contents": {"text": {"maxCharacters": 1200}},
        }
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{self.base_url}/search",
                headers={"x-api-key": self.api_key, "content-type": "application/json"},
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        sources: list[RetrievedSource] = []
        for item in data.get("results", []):
            text = item.get("text") or ""
            highlights = item.get("highlights") or []
            snippet = text or (highlights[0] if highlights else "")
            if not item.get("url"):
                continue
            sources.append(
                RetrievedSource(
                    title=item.get("title"),
                    url=item["url"],
                    snippet=snippet[:1200],
                    published_at=item.get("publishedDate"),
                    score=item.get("score"),
                )
            )
        return sources

    async def retrieve(self, event: ProphetEvent) -> list[RetrievedSource]:
        collected: list[RetrievedSource] = []
        seen_urls: set[str] = set()
        for round_index in range(self.rounds):
            for query in build_round_queries(event, round_index):
                for source in await self._search_one(query):
                    if source.url in seen_urls:
                        continue
                    seen_urls.add(source.url)
                    collected.append(source)
                    if len(collected) >= self.max_sources:
                        return collected
        return collected


def render_evidence(sources: list[RetrievedSource]) -> str:
    if not sources:
        return "No external retrieval evidence available."
    lines = []
    for idx, source in enumerate(sources, start=1):
        lines.append(
            f"{idx}. {source.title or 'Untitled'} | {source.url}\n"
            f"Published: {source.published_at or 'unknown'} | Score: {source.score or 'n/a'}\n"
            f"{source.snippet}"
        )
    return "\n\n".join(lines)
