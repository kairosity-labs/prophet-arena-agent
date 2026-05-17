from __future__ import annotations

from prophet_arena_agent.models import ProphetEvent
from prophet_arena_agent.retrieval import build_round_queries


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
