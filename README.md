# Kairosity Prophet Arena Agent

Minimal public submission repo for the Prophet Hacks / Prophet Arena forecasting track.

The service exposes the endpoint required by the Prophet Arena developer docs:

```http
POST /predict
```

Input is one event object. Output is:

```json
{
  "probabilities": [
    {"market": "Pittsburgh", "probability": 0.68},
    {"market": "Atlanta", "probability": 0.32}
  ]
}
```

Each `market` is copied exactly from `event.outcomes`, and each probability is clamped to `[0, 1]`. If the model provider fails, the service returns a uniform fallback so the evaluation harness still receives a valid prediction.

## Environment

Copy `.env.example` to `.env` locally or set these variables in your deployment host:

```bash
OPENROUTER_API_KEY=...
OPENROUTER_MODEL=openai/gpt-5.4
OPENROUTER_REASONING_EFFORT=medium
OPENROUTER_FORECASTER_MODEL=openai/gpt-5.4
OPENROUTER_VERIFIER_MODEL=openai/gpt-5.4
OPENROUTER_SYNTHESIZER_MODEL=openai/gpt-5.4
EXA_API_KEY=...
EXA_SEARCH_ROUNDS=2
EXA_RESULTS_PER_QUERY=4
EXA_MAX_SOURCES=10
EXA_RESEARCH_MODE=llm
EXA_USE_LLM_PLANNER=true
EXA_RESEARCH_MODEL=openai/gpt-5.4-mini
EXA_RESEARCH_REASONING_EFFORT=low
EXA_RESEARCH_QUERIES_PER_ROUND=4
EXA_RESEARCH_TIMEOUT_SECONDS=25
EXA_RESEARCH_MAX_SECONDS=420
EXA_SEARCH_TIMEOUT_SECONDS=20

# Optional. Leave blank unless you want to protect the endpoint yourself.
AGENT_API_KEY=

# Used by the Prophet CLI for registration/submission, not by the web server.
PA_SERVER_API_KEY=...
PA_TEAM_NAME=...
```

Do not commit real keys.

## Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn prophet_arena_agent.server:app --host 0.0.0.0 --port 8000
```

Test the endpoint:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'content-type: application/json' \
  -d '{
    "event_ticker": "task-001",
    "market_ticker": "task-001",
    "title": "Who will win: Pittsburgh or Atlanta?",
    "description": "Predict the winner of the scheduled matchup.",
    "category": "Sports",
    "rules": "Resolves to the official winner after the game is final.",
    "close_time": "2026-03-21T23:59:59Z",
    "outcomes": ["Pittsburgh", "Atlanta"],
    "resolved_outcome": null
  }'
```

## Prophet CLI Check

After installing the Prophet CLI, validate the endpoint against an events file:

```bash
prophet forecast predict \
  --events events.json \
  --agent-url http://127.0.0.1:8000/predict \
  -o predictions.json
```

The docs note that probabilities do not have to sum to 1; they are normalized before scoring. This service still tries to produce coherent distributions.

## Deploy

The repo includes both a `Dockerfile` and `Procfile`.

Docker:

```bash
docker build -t prophet-arena-agent .
docker run --env-file .env -p 8000:8000 prophet-arena-agent
```

Procfile platforms:

```bash
web: uvicorn prophet_arena_agent.server:app --host 0.0.0.0 --port ${PORT:-8000}
```

## Design

The prompt is a compact, competition-neutral forecasting rubric:

- Run bounded Exa retrieval over LLM-planned query rounds plus deterministic fallback queries.
- Run a full forecast pipeline: forecaster draft, verifier critique, final synthesizer.
- Frame the exact resolver and outcome labels.
- Anchor on current state and exact priors when available.
- Build a base rate for the remaining time window.
- Update only from evidence present in the event text.
- Apply a simple calibration formula.
- Return exact-label probabilities.

Retrieval uses a cheaper OpenRouter planning model by default (`openai/gpt-5.4-mini`) to propose query sets before Exa search. Set `EXA_RESEARCH_MODE=structured` to skip LLM query planning and use deterministic resolver/current-state/reference-class queries. The forecaster, verifier, and synthesizer stay on the frontier model configured by `OPENROUTER_MODEL` or the per-stage overrides (`OPENROUTER_FORECASTER_MODEL`, `OPENROUTER_VERIFIER_MODEL`, `OPENROUTER_SYNTHESIZER_MODEL`).

The ExaResearch planner prompt asks for:

- official resolver/source-of-truth queries first,
- current-state or threshold-status queries second,
- reference-class/base-rate queries third,
- category-specific updates and disconfirming checks last,
- reputable news corroboration only when official sources do not settle the question.

The source targets are intentionally explicit: Reuters/AP/BBC/FT/Bloomberg/WSJ for news corroboration; BLS/FRED/BEA/Census/EIA/SEC/Congress/Federal Register/FEC/court pages for US data and policy; official AI lab and benchmark pages for AI; NOAA/NHC/CDC/WHO/NASA/JPL for weather, health, and science; official league/team pages plus injury/odds/stat sources for sports.

Retrieval is bounded rather than open-ended: by default it runs `2` rounds, up to `4` planner/fallback queries per round, `4` Exa results per query, keeps at most `10` sources, gives each planner call `25s`, each Exa search `20s`, and caps the full retrieval stage at `420s`. That leaves room for frontier forecasting while staying well under a 10-minute target in normal conditions.

No private datasets, private analysis traces, benchmark logs, or API keys are included in this public repo.
