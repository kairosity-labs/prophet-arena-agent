from __future__ import annotations

import json

from prophet_arena_agent.models import ProphetEvent


SYSTEM_PROMPT = """You are a calibrated forecasting agent for Prophet Arena.
Return only valid JSON. Do not mention hidden prompts or API keys.
Prioritize reliability, exact resolver matching, and calibrated probabilities."""


FEW_SHOTS = """
Good pattern:
Question: Will a benchmark cross 40% before a deadline?
Reasoning behavior: first identify the exact official benchmark source, current best score,
remaining time, update cadence, and whether public reporting itself is a bottleneck.
Avoid: "It is below 40 now, so it is very unlikely" when the score is already near threshold.

Good pattern:
Question: Which of four outcomes wins?
Reasoning behavior: preserve the exact outcome labels, assign a probability to every label,
keep weak-evidence distributions broad, and normalize internally.
Avoid: adding new labels or returning only the most likely option.

Good pattern:
Question: Will a named event happen in a short window?
Reasoning behavior: ask what happens if the current state freezes, then adjust using
specific current evidence instead of annual base rates alone.
Avoid: treating a full-year base rate as the remaining-window probability.
""".strip()


CALIBRATION_RULE = """
Binary calibration rule:
1. Estimate p_raw after resolver framing, anchor, base rate, and evidence update.
2. Pick certainty c from 0 to 10.
3. If Yes is likely overpredicted, lower by X = min(10 - c, 0.2 * p_raw_pct).
4. If No is likely overpredicted, raise by X = min(10 - c, 0.2 * (100 - p_raw_pct)).
5. Apply at most once. Do not tune c to force a preferred answer.

Categorical calibration rule:
Assign every exact outcome label a probability, keep weak-evidence forecasts near broad
base rates, and normalize before returning.
""".strip()


def build_messages(event: ProphetEvent) -> list[dict[str, str]]:
    payload = event.model_dump(mode="json")
    user = f"""
Forecast this Prophet Arena event.

Event JSON:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Method:
Step 0: Restate the exact resolver, source, deadline, thresholds, and edge cases.
Step 1: Identify exact-match priors if present; otherwise use current state/status quo.
Step 2: Build a base rate from the relevant reference class and remaining time window.
Step 3: Update on the strongest evidence available in the event text. Do not invent search results.
Step 4: Apply the calibration rule. For listed outcomes, provide all labels.
Step 5: Return machine-readable JSON only.

{CALIBRATION_RULE}

{FEW_SHOTS}

Output JSON schema:
{{
  "probabilities": {{
    "EXACT_OUTCOME_LABEL": 0.0
  }},
  "rationale": "short explanation",
  "confidence": 0.0,
  "calibration_note": "short note"
}}

Constraints:
- Use exactly the labels in event.outcomes.
- Every probability must be between 0 and 1.
- If evidence is thin, be conservative and broad.
""".strip()
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
