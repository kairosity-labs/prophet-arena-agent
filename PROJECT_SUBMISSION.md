# Project Name

Kairosity Prophet Arena Agent

# Elevator Pitch

A small, reliable forecasting endpoint that turns Prophet Arena event JSON into calibrated probability forecasts with exact outcome labels.

# About The Project

Kairosity Prophet Arena Agent was inspired by the core challenge of Prophet Hacks: forecasting systems are only useful when they are both thoughtful and operationally reliable. The endpoint therefore optimizes for two things at once: a disciplined forecast workflow and a strict API contract that always returns valid probabilities.

The agent uses a compact stepwise forecasting rubric:

1. Read the resolver carefully.
2. Preserve the exact event outcome labels.
3. Anchor on current state and relevant priors.
4. Build a base rate for the remaining time window.
5. Apply evidence updates and calibration.
6. Return machine-readable probabilities.

The calibration rule for binary questions is intentionally simple:

$$
X = \\min(10 - c, 0.2p)
$$

where \(p\) is the raw percentage estimate and \(c\) is confidence on a 0-10 scale. The agent applies this idea carefully: it avoids mechanical over-adjustment, and for categorical questions it instead returns a full normalized probability vector over every exact label.

The biggest lesson was that forecasting quality is not just model quality. The surrounding contract matters: exact labels, sane fallbacks, timeout safety, and provider failure handling are part of the forecasting system. A beautiful forecast that misses the schema is still a failed forecast.

The main challenge was keeping the public submission small while preserving the most useful parts of a larger agentic forecasting architecture. The final repo intentionally omits private analysis artifacts and keeps only the deployable endpoint, prompt, tests, and operational instructions.

# Built With

- Python
- FastAPI
- Pydantic
- Uvicorn
- HTTPX
- OpenAI API
- GPT-5.5 / GPT-5.4-family OpenAI models
- Docker
- Prophet Arena / Prophet Hacks forecasting endpoint contract
- Pytest
- Ruff
