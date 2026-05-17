FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY prophet_arena_agent ./prophet_arena_agent

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "prophet_arena_agent.server:app", "--host", "0.0.0.0", "--port", "8000"]
