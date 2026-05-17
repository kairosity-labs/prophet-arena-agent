from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx

from prophet_arena_agent.models import ProphetEvent


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

EXA_RESEARCH_SYSTEM = """
You are ExaResearch, a bounded research-query planner for a forecasting agent.
Your only job is to propose high-yield Exa web-search queries. Do not forecast.

The strongest forecasting traces use retrieval in this order:
1. Exact resolver first: official source, scoring rule, threshold, deadline, and edge cases.
2. Exact-match/current-state evidence: the live value or status that directly determines the
   event, plus exact-match crowd/market priors if available.
3. Reference classes: historical frequencies, same-source base rates, similar thresholds,
   same tournament/category analogs, and time-window rates.
4. Update evidence: catalysts, recent changes, scheduled events, injuries, releases,
   filings, legislative status, polling, weather outlooks, or official announcements.
5. Disconfirming checks: evidence that would make the headline story misleading.

Derive queries by decomposing the event into:
- resolver source: Who grades this, and what page/data feed will settle it?
- subject entity: company, team, person, bill, model, index, country, dataset, benchmark.
- threshold and unit: number, rank, score, date, price, release, count, or status.
- time horizon: remaining window, publication lag, event date, season/month/year.
- reference class: same entity, same source, same month/season, same threshold family.

Diversify by category:
- AI/technology: official lab/product pages, model-release announcements, benchmark
  leaderboards, eval rules, changelogs, safety/system cards, developer docs.
- Finance/markets: ticker/current price, official exchange data, SEC filings, earnings
  calendar, investor relations, central-bank calendar, macro release schedule.
- Politics/law/policy: official bill or docket page, government press releases, legislative
  calendars, polling averages, court dockets, regulator notices.
- Sports: official schedule/result source, injury reports, rosters, form tables, bookmaker
  odds, historical same-matchup/base-rate stats.
- Weather/climate/public health: NOAA/NHC/WHO/CDC/ministry source pages, surveillance data,
  seasonal outlooks, historical same-month rates, official advisories.
- Geopolitics/security: official statements, sanctions pages, treaty/agency pages, conflict
  trackers, credible local/international reporting for corroboration.
- Science/space: official mission/status pages, NASA/ESA/JPL data, observatory pages,
  launch manifests, historical event-rate datasets.

Source/API targets to prefer when relevant:
- General news corroboration: Reuters, Associated Press, BBC, Financial Times, Bloomberg,
  Wall Street Journal, The Economist, Politico, Axios, The Verge, TechCrunch. Use these for
  updates and corroboration, not as the resolver if an official source exists.
- US data/API sources: bls.gov, fred.stlouisfed.org, bea.gov, census.gov, eia.gov,
  sec.gov/edgar, congress.gov, federalregister.gov, fec.gov, uscourts.gov.
- Markets/company sources: exchange pages, investor-relations pages, SEC filings, earnings
  calendars, central-bank calendars, CME/FRED pages, and official index-provider pages.
- AI/benchmark sources: openai.com/news, anthropic.com/news, deepmind.google, ai.meta.com,
  x.ai/news, epoch.ai, benchmark/eval leaderboards, official GitHub/changelog pages.
- Weather/health/science APIs: weather.gov, nhc.noaa.gov, climate.gov, cdc.gov, who.int,
  data.nasa.gov, cneos.jpl.nasa.gov, esa.int, arxiv.org only for research context.
- Sports sources: official league/team pages, injury reports, schedules/results, reliable
  odds aggregators, sports-reference/stathead-style historical pages.

Compact good/bad query patterns:
- AI good: "site:openai.com/news GPT-5.5 release date official" plus "Epoch AI FrontierMath
  leaderboard 45% model October 2026". Bad: "new AI model rumors".
- Finance good: "AAPL investor relations Q2 2026 guidance official" plus "SEC EDGAR AAPL
  10-Q revenue segment". Bad: "AAPL stock vibes".
- Politics good: "congress.gov HR 1234 latest action 2026" plus "committee calendar HR 1234".
  Bad: "will bill pass news".
- Sports good: "NBA official injury report Lakers Celtics March 2026" plus "Lakers Celtics
  head to head closing odds". Bad: "who will win Lakers Celtics".
- Weather/health good: "NOAA March 2026 global temperature report warmest March" plus
  "NOAA March warmest March historical record". Bad: "March was hot news".
- Geopolitics good: "official sanctions notice country entity date" plus "Reuters AP
  corroboration sanctions entity". Bad: "geopolitical tensions latest".

Round behavior:
- Round 0 should cover official resolver/current state plus one broad reference-class query.
- Later rounds should read the evidence digest, identify missing pieces, and ask targeted
  follow-up queries rather than repeating broad searches.

Return JSON only:
{"queries": ["query 1", "query 2"], "rationale": "one short sentence"}

Constraints:
- Produce at most the requested number of queries.
- Keep each query under 18 words when possible.
- Include exact names, dates, thresholds, and official-source hints when known.
- Do not invent URLs. Do not include private reasoning. Do not return prose outside JSON.
""".strip()


@dataclass(frozen=True)
class RetrievedSource:
    title: str | None
    url: str
    snippet: str
    published_at: str | None = None
    score: float | None = None


def _dedupe_queries(queries: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for query in queries:
        cleaned = _clean_query(query)
        if cleaned and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            deduped.append(cleaned)
    return deduped


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

    return _dedupe_queries(queries)


def _extract_json_object(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    first = text.find("{")
    last = text.rfind("}")
    if first >= 0 and last > first:
        parsed = json.loads(text[first : last + 1])
        if isinstance(parsed, dict):
            return parsed
    return {}


def parse_planned_queries(text: str, max_queries: int) -> list[str]:
    data = _extract_json_object(text)
    raw_queries = data.get("queries", [])
    if not isinstance(raw_queries, list):
        return []
    queries = [query for query in raw_queries if isinstance(query, str)]
    return _dedupe_queries(queries)[:max(1, max_queries)]


def _render_source_digest(sources: list[RetrievedSource]) -> str:
    if not sources:
        return "No prior retrieved sources in this question."
    lines: list[str] = []
    for idx, source in enumerate(sources[:8], start=1):
        snippet = _clean_query(source.snippet[:360])
        lines.append(
            f"{idx}. {source.title or 'Untitled'} | {source.url}\n"
            f"Published: {source.published_at or 'unknown'}\n"
            f"Snippet: {snippet}"
        )
    return "\n\n".join(lines)


def build_research_planner_messages(
    event: ProphetEvent,
    *,
    round_index: int,
    previous_sources: list[RetrievedSource],
    fallback_queries: list[str],
    max_queries: int,
) -> list[dict[str, str]]:
    payload = event.model_dump(mode="json")
    user = f"""
Plan Exa search queries for this forecasting event.

Round index: {round_index}
Maximum queries to return: {max_queries}

Event JSON:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Previous evidence digest:
{_render_source_digest(previous_sources)}

Deterministic fallback queries, for orientation only:
{json.dumps(fallback_queries, ensure_ascii=False, indent=2)}

Query-planning checklist:
1. Ask at least one query that targets the official resolver or source of truth.
2. Ask at least one query that would reveal the current status/value relative to the threshold.
3. Ask at least one reference-class or base-rate query when the event is not already settled.
4. Diversify the remaining query by category-specific evidence or disconfirming checks.
5. On later rounds, avoid duplicate searches and fill the biggest evidence gap.
""".strip()
    return [
        {"role": "system", "content": EXA_RESEARCH_SYSTEM},
        {"role": "user", "content": user},
    ]


class ExaQueryPlanner:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        reasoning_effort: str = "low",
        timeout_seconds: float = 25.0,
        max_queries: int = 4,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.timeout_seconds = max(1.0, timeout_seconds)
        self.max_queries = max(1, max_queries)

    @classmethod
    def from_env(cls) -> "ExaQueryPlanner | None":
        legacy_enabled = os.environ.get("EXA_USE_LLM_PLANNER", "true").lower()
        research_mode = os.environ.get(
            "EXA_RESEARCH_MODE",
            "llm" if legacy_enabled not in {"0", "false", "no", "off"} else "structured",
        ).lower()
        if research_mode == "structured" or legacy_enabled in {"0", "false", "no", "off"}:
            return None
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            return None
        return cls(
            api_key=api_key,
            model=os.environ.get("EXA_RESEARCH_MODEL")
            or os.environ.get("OPENROUTER_MODEL", "openai/gpt-5.4-mini"),
            reasoning_effort=os.environ.get("EXA_RESEARCH_REASONING_EFFORT", "low"),
            timeout_seconds=float(os.environ.get("EXA_RESEARCH_TIMEOUT_SECONDS", "25")),
            max_queries=int(os.environ.get("EXA_RESEARCH_QUERIES_PER_ROUND", "4")),
        )

    async def plan_queries(
        self,
        event: ProphetEvent,
        *,
        round_index: int,
        previous_sources: list[RetrievedSource],
        fallback_queries: list[str],
    ) -> list[str]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": build_research_planner_messages(
                event,
                round_index=round_index,
                previous_sources=previous_sources,
                fallback_queries=fallback_queries,
                max_queries=self.max_queries,
            ),
            "response_format": {"type": "json_object"},
        }
        if self.reasoning_effort and self.reasoning_effort.lower() != "none":
            payload["reasoning"] = {"effort": self.reasoning_effort}

        headers = {
            "authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
            "http-referer": "https://github.com/kairosity-labs/prophet-arena-agent",
            "x-title": "Kairosity Prophet Arena ExaResearch",
        }

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(OPENROUTER_URL, headers=headers, json=payload)
            if response.status_code == 400 and "reasoning" in payload:
                payload.pop("reasoning", None)
                response = await client.post(OPENROUTER_URL, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return parse_planned_queries(content, self.max_queries)


class ExaRetriever:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.exa.ai",
        rounds: int = 2,
        results_per_query: int = 4,
        max_sources: int = 10,
        query_planner: Any | None = None,
        queries_per_round: int = 4,
        search_timeout_seconds: float = 20.0,
        max_elapsed_seconds: float = 420.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.rounds = max(0, rounds)
        self.results_per_query = max(1, results_per_query)
        self.max_sources = max(1, max_sources)
        self.query_planner = query_planner
        self.queries_per_round = max(1, queries_per_round)
        self.search_timeout_seconds = max(1.0, search_timeout_seconds)
        self.max_elapsed_seconds = max(1.0, max_elapsed_seconds)

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
            query_planner=ExaQueryPlanner.from_env(),
            queries_per_round=int(os.environ.get("EXA_RESEARCH_QUERIES_PER_ROUND", "4")),
            search_timeout_seconds=float(os.environ.get("EXA_SEARCH_TIMEOUT_SECONDS", "20")),
            max_elapsed_seconds=float(os.environ.get("EXA_RESEARCH_MAX_SECONDS", "420")),
        )

    async def _search_one(self, query: str) -> list[RetrievedSource]:
        payload: dict[str, Any] = {
            "query": query,
            "numResults": self.results_per_query,
            "contents": {"text": {"maxCharacters": 1200}},
        }
        async with httpx.AsyncClient(timeout=self.search_timeout_seconds) as client:
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

    async def _round_queries(
        self,
        event: ProphetEvent,
        *,
        round_index: int,
        previous_sources: list[RetrievedSource],
    ) -> list[str]:
        fallback_queries = build_round_queries(event, round_index)
        planned_queries: list[str] = []
        if self.query_planner:
            try:
                planned_queries = await self.query_planner.plan_queries(
                    event,
                    round_index=round_index,
                    previous_sources=previous_sources,
                    fallback_queries=fallback_queries,
                )
            except Exception:
                planned_queries = []
        return _dedupe_queries(planned_queries + fallback_queries)[: self.queries_per_round]

    async def retrieve(self, event: ProphetEvent) -> list[RetrievedSource]:
        collected: list[RetrievedSource] = []
        seen_urls: set[str] = set()
        deadline = time.monotonic() + self.max_elapsed_seconds
        for round_index in range(self.rounds):
            if time.monotonic() >= deadline:
                return collected
            queries = await self._round_queries(
                event,
                round_index=round_index,
                previous_sources=collected,
            )
            for query in queries:
                if time.monotonic() >= deadline:
                    return collected
                try:
                    sources = await self._search_one(query)
                except Exception:
                    continue
                for source in sources:
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
