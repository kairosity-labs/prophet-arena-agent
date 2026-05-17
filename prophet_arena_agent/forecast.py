from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import httpx

from prophet_arena_agent.models import ForecastJSON, Prediction, Probability, ProphetEvent
from prophet_arena_agent.prompt import (
    build_forecaster_messages,
    build_synthesizer_messages,
    build_verifier_messages,
)
from prophet_arena_agent.retrieval import ExaRetriever, render_evidence


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


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

    if sum(item.probability for item in cleaned) <= 0:
        return uniform_prediction(event)
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
    stage_key = f"OPENROUTER_{stage.upper()}_MODEL"
    return os.environ.get(stage_key) or os.environ.get("OPENROUTER_MODEL", "openai/gpt-5.4")


def _stage_reasoning_effort(stage: str) -> str:
    stage_key = f"OPENROUTER_{stage.upper()}_REASONING_EFFORT"
    return os.environ.get(stage_key) or os.environ.get("OPENROUTER_REASONING_EFFORT", "medium")


def _forecast_branch_count() -> int:
    try:
        return max(1, int(os.environ.get("FORECAST_BRANCH_COUNT", "2")))
    except ValueError:
        return 2


async def call_openrouter_messages(messages: list[dict[str, str]], *, stage: str) -> dict[str, Any]:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    model = _stage_model(stage)
    reasoning_effort = _stage_reasoning_effort(stage)
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
    }
    if reasoning_effort and reasoning_effort.lower() != "none":
        payload["reasoning"] = {"effort": reasoning_effort}

    headers = {
        "authorization": f"Bearer {api_key}",
        "content-type": "application/json",
        "http-referer": "https://github.com/kairosity-labs/prophet-arena-agent",
        "x-title": "Kairosity Prophet Arena Agent",
    }

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(OPENROUTER_URL, headers=headers, json=payload)
        if response.status_code == 400 and "reasoning" in payload:
            payload.pop("reasoning", None)
            response = await client.post(OPENROUTER_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return extract_json(content)


async def _run_forecaster_verifier_branch(
    event: ProphetEvent,
    *,
    evidence: str,
    branch: int,
) -> dict[str, Any]:
    forecast_json = await call_openrouter_messages(
        build_forecaster_messages(event, evidence=evidence),
        stage="forecaster",
    )
    verifier_json = await call_openrouter_messages(
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
    return await call_openrouter_messages(
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
