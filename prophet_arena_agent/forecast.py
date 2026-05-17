from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from prophet_arena_agent.models import ForecastJSON, Prediction, Probability, ProphetEvent
from prophet_arena_agent.prompt import (
    build_forecaster_messages,
    build_synthesizer_messages,
    build_verifier_messages,
)
from prophet_arena_agent.retrieval import ExaRetriever, render_evidence


def uniform_prediction(event: ProphetEvent) -> Prediction:
    outcomes = event.outcomes or ["Yes", "No"]
    probability = 1.0 / len(outcomes)
    return Prediction(
        probabilities=[
            Probability(market=outcome, probability=probability)
            for outcome in outcomes
        ]
    )


def prediction_from_model_json(event: ProphetEvent, data: dict[str, Any]) -> Prediction:
    forecast = ForecastJSON.model_validate(data)
    outcomes = event.outcomes or ["Yes", "No"]
    by_label = forecast.probabilities or {}

    if not by_label and forecast.p_yes is not None:
        lowered = {outcome.lower(): outcome for outcome in outcomes}
        if "yes" in lowered and "no" in lowered:
            by_label = {
                lowered["yes"]: forecast.p_yes,
                lowered["no"]: 1.0 - forecast.p_yes,
            }

    if not by_label:
        return uniform_prediction(event)

    cleaned: list[Probability] = []
    for outcome in outcomes:
        value = by_label.get(outcome)
        if value is None:
            value = by_label.get(outcome.strip())
        if value is None:
            value = 0.0
        cleaned.append(Probability(market=outcome, probability=max(0.0, min(1.0, float(value)))))

    return Prediction(probabilities=cleaned)


def extract_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    first = text.find("{")
    last = text.rfind("}")
    if first >= 0 and last > first:
        return json.loads(text[first : last + 1])
    raise ValueError("Model did not return a JSON object")


def _stage_model(stage: str) -> str:
    stage_key = f"OPENAI_{stage.upper()}_MODEL"
    model = os.environ.get(stage_key) or os.environ.get("OPENAI_MODEL", "gpt-5.5")
    return model.removeprefix("openai/")


def _stage_reasoning_effort(stage: str) -> str:
    stage_key = f"OPENAI_{stage.upper()}_REASONING_EFFORT"
    return os.environ.get(stage_key) or os.environ.get("OPENAI_REASONING_EFFORT", "medium")


def _uses_reasoning_controls(model: str) -> bool:
    return model.startswith(("gpt-5", "o"))


def _forecast_branch_count() -> int:
    try:
        return max(1, int(os.environ.get("FORECAST_BRANCH_COUNT", "2")))
    except ValueError:
        return 2


async def call_openai_messages(messages: list[dict[str, str]], *, stage: str) -> dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError("Install the openai package to use the default backend") from exc

    model = _stage_model(stage)
    reasoning_effort = _stage_reasoning_effort(stage)
    payload: dict[str, Any] = {
        "model": model,
        "input": messages,
    }
    if reasoning_effort and reasoning_effort.lower() != "none" and _uses_reasoning_controls(model):
        payload["reasoning"] = {"effort": reasoning_effort}
    client = AsyncOpenAI(api_key=api_key, timeout=120)
    try:
        response = await client.responses.create(**payload)
    except Exception:
        if "reasoning" not in payload:
            raise
        payload.pop("reasoning", None)
        response = await client.responses.create(**payload)

    content = getattr(response, "output_text", None) or str(response)
    return extract_json(content)


async def _run_forecaster_verifier_branch(
    event: ProphetEvent,
    *,
    evidence: str,
    branch: int,
) -> dict[str, Any]:
    forecast_json = await call_openai_messages(
        build_forecaster_messages(event, evidence=evidence),
        stage="forecaster",
    )
    verifier_json = await call_openai_messages(
        build_verifier_messages(event, evidence=evidence, forecast_json=forecast_json),
        stage="verifier",
    )
    return {"branch": branch, "forecast": forecast_json, "verifier": verifier_json}


async def run_forecast_pipeline(event: ProphetEvent, evidence: str) -> dict[str, Any]:
    branches = await asyncio.gather(
        *[
            _run_forecaster_verifier_branch(event, evidence=evidence, branch=idx + 1)
            for idx in range(_forecast_branch_count())
        ]
    )
    return await call_openai_messages(
        build_synthesizer_messages(
            event,
            evidence=evidence,
            forecast_json=[branch["forecast"] for branch in branches],
            verifier_json=[
                {
                    "branch": branch["branch"],
                    "verifier": branch["verifier"],
                    "forecast": branch["forecast"],
                }
                for branch in branches
            ],
        ),
        stage="synthesizer",
    )


async def predict_event(event: ProphetEvent) -> Prediction:
    try:
        retriever = ExaRetriever.from_env()
        sources = await retriever.retrieve(event) if retriever else []
        model_json = await run_forecast_pipeline(event, render_evidence(sources))
        return prediction_from_model_json(event, model_json)
    except Exception:
        # Completion rate matters in Prophet Arena. On provider failures, return a safe fallback.
        return uniform_prediction(event)
