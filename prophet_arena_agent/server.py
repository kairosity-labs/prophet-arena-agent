from __future__ import annotations

import os
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException

from prophet_arena_agent.forecast import predict_event
from prophet_arena_agent.models import Prediction, ProphetEvent

load_dotenv(override=False)

app = FastAPI(title="Kairosity Prophet Arena Agent", version="0.1.0")


def check_optional_auth(authorization: str | None, x_api_key: str | None) -> None:
    expected = os.environ.get("AGENT_API_KEY")
    if not expected:
        return
    bearer = f"Bearer {expected}"
    if authorization == bearer or x_api_key == expected:
        return
    raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/")
async def root() -> dict[str, str]:
    return {"ok": "true", "service": "prophet-arena-agent"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"ok": "true"}


@app.post("/predict", response_model=Prediction)
async def predict(
    event: ProphetEvent,
    authorization: Annotated[str | None, Header()] = None,
    x_api_key: Annotated[str | None, Header()] = None,
) -> Prediction:
    check_optional_auth(authorization, x_api_key)
    return await predict_event(event)


def main() -> None:
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("prophet_arena_agent.server:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
