from __future__ import annotations

from prophet_arena_agent.models import ProphetEvent
from prophet_arena_agent.retrieval import (
    EXA_RESEARCH_SYSTEM,
    build_research_planner_messages,
    build_round_queries,
    parse_planned_queries,
)


def test_retrieval_queries_are_bounded_and_relevant() -> None:
    event = ProphetEvent(
        title="Will a major AI lab release a model before July?",
        category="Technology",
        rules="Resolves Yes if a public release happens before the deadline.",
        outcomes=["Yes", "No"],
    )

    round_0 = build_round_queries(event, 0)
    round_1 = build_round_queries(event, 1)

    assert 1 <= len(round_0) <= 2
    assert 1 <= len(round_1) <= 3
    assert "major AI lab" in round_0[0]
    assert any("latest" in query for query in round_1)


def test_exa_research_prompt_contains_category_sources_and_examples() -> None:
    assert "official resolver" in EXA_RESEARCH_SYSTEM
    assert "Reference classes" in EXA_RESEARCH_SYSTEM
    assert "Reuters" in EXA_RESEARCH_SYSTEM
    assert "sec.gov/edgar" in EXA_RESEARCH_SYSTEM
    assert "NOAA" in EXA_RESEARCH_SYSTEM
    assert "AI good" in EXA_RESEARCH_SYSTEM
    assert "Bad:" in EXA_RESEARCH_SYSTEM


def test_research_planner_messages_include_fallback_queries() -> None:
    event = ProphetEvent(
        title="Will NOAA report March 2026 as the warmest March?",
        category="Weather",
        rules="Resolves using the official NOAA global climate report.",
        outcomes=["Yes", "No"],
    )
    messages = build_research_planner_messages(
        event,
        round_index=1,
        previous_sources=[],
        fallback_queries=["NOAA March 2026 official report"],
        max_queries=4,
    )

    assert messages[0]["role"] == "system"
    assert "NOAA March 2026 official report" in messages[1]["content"]
    assert "reference-class" in messages[1]["content"]


def test_parse_planned_queries_dedupes_and_bounds() -> None:
    parsed = parse_planned_queries(
        '{"queries":["NOAA March 2026 report"," NOAA March 2026 report ",'
        '"NOAA warmest March historical base rate"]}',
        max_queries=1,
    )

    assert parsed == ["NOAA March 2026 report"]
