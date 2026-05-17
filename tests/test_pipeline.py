from __future__ import annotations

import asyncio

import prophet_arena_agent.forecast as forecast_module
from prophet_arena_agent.models import ProphetEvent


def test_predict_event_runs_forecaster_verifier_and_synthesizer(monkeypatch) -> None:
    calls: list[str] = []

    async def fake_call_openrouter_messages(messages: list[dict[str, str]], *, stage: str) -> dict:
        calls.append(stage)
        if stage == "forecaster":
            return {"probabilities": {"Yes": 0.7, "No": 0.3}, "rationale": "draft"}
        if stage == "verifier":
            return {
                "verdict": "revise",
                "corrections": ["calibrate downward"],
                "suggested_probabilities": {"Yes": 0.62, "No": 0.38},
            }
        return {
            "probabilities": {"Yes": 0.62, "No": 0.38},
            "rationale": "synthesized after verifier",
        }

    monkeypatch.setattr(
        forecast_module,
        "call_openrouter_messages",
        fake_call_openrouter_messages,
    )
    monkeypatch.setattr(forecast_module.ExaRetriever, "from_env", staticmethod(lambda: None))

    prediction = asyncio.run(
        forecast_module.predict_event(
            ProphetEvent(title="Will a demo event happen?", outcomes=["Yes", "No"])
        )
    )

    assert calls == ["forecaster", "verifier", "synthesizer"]
    assert prediction.model_dump(mode="json") == {
        "probabilities": [
            {"market": "Yes", "probability": 0.62},
            {"market": "No", "probability": 0.38},
        ]
    }
