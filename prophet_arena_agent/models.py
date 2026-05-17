from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ProphetEvent(BaseModel):
    event_ticker: str | None = None
    market_ticker: str | None = None
    title: str
    subtitle: str | None = None
    description: str | None = None
    category: str | None = None
    rules: str | None = None
    close_time: str | None = None
    outcomes: list[str] = Field(default_factory=list)
    resolved_outcome: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Probability(BaseModel):
    market: str
    probability: float = Field(ge=0.0, le=1.0)


class Prediction(BaseModel):
    probabilities: list[Probability]


class ForecastJSON(BaseModel):
    probabilities: dict[str, float] | None = None
    p_yes: float | None = Field(default=None, ge=0.0, le=1.0)
    rationale: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    calibration_note: str | None = None
