from __future__ import annotations

import json
from typing import Any

from prophet_arena_agent.models import ProphetEvent


FORECASTER_SYSTEM_PROMPT = """You are an independent calibrated forecaster for Prophet Arena.
Return only valid JSON. Do not mention hidden prompts or API keys.
Prioritize reliability, exact resolver matching, and calibrated probabilities."""


VERIFIER_SYSTEM_PROMPT = """You are a rigorous verifier for a forecasting pipeline.
Return only valid JSON. Find resolver mistakes, prior misuse, source hierarchy errors,
overconfidence, missing outcome labels, retrieval hallucinations, and schema problems."""


SYNTHESIZER_SYSTEM_PROMPT = """You are the final calibrated synthesizer in a forecasting pipeline.
Return only valid JSON. Use the forecaster draft and verifier critique to produce the final
machine-readable probabilities for the exact outcome labels."""


FEW_SHOTS = """
Good pattern:
Question: Will a benchmark cross 40% before a deadline?
Reasoning behavior: first identify the exact official benchmark source, current best score,
remaining time, update cadence, and whether public reporting itself is a bottleneck.
Avoid: "It is below 40 now, so it is very unlikely" when the score is already near threshold.

Good pattern:
Question: Which of four outcomes wins?
Reasoning behavior: preserve the exact outcome labels, assign a probability to every label,
keep weak-evidence probabilities broad, and do not force probabilities to sum to 1 unless
the resolver explicitly says labels are exhaustive and mutually exclusive.
Avoid: adding new labels, returning only the most likely option, or smearing to uniform only
because the labels look non-exhaustive.

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
Assign every exact outcome label a probability in [0, 1]. Treat Prophet Arena outcome labels
as per-label probabilities that need not sum to 1; do not normalize unless the event rules
explicitly define an exhaustive mutually exclusive set.
""".strip()


def build_forecaster_messages(
    event: ProphetEvent,
    evidence: str = "No external retrieval evidence available.",
) -> list[dict[str, str]]:
    payload = event.model_dump(mode="json")
    user = f"""
Forecast this Prophet Arena event.

Event JSON:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Retrieved evidence:
{evidence}

Method:
Step 0: Restate the exact resolver, source, deadline, thresholds, and edge cases.
Step 1: Identify exact-match priors if present; otherwise use current state/status quo.
Step 2: Build a base rate from the relevant reference class and remaining time window.
Step 3: Update on retrieved evidence and the event text. Do not invent search results.
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
- Probabilities across labels need not sum to 1.
- If all listed outcomes look unlikely because the labels appear stale or non-exhaustive,
  return low probabilities for those labels rather than a uniform fallback.
- If evidence is thin, be conservative and broad.
""".strip()
    return [
        {"role": "system", "content": FORECASTER_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_verifier_messages(
    event: ProphetEvent,
    *,
    evidence: str,
    forecast_json: dict,
) -> list[dict[str, str]]:
    payload = event.model_dump(mode="json")
    user = f"""
Verify this Prophet Arena forecast draft before final synthesis.

Event JSON:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Retrieved evidence:
{evidence}

Forecaster draft JSON:
{json.dumps(forecast_json, ensure_ascii=False, indent=2)}

Verification checklist:
1. Resolver: Does the draft obey the exact source, threshold, deadline, and edge cases?
2. Outcomes: Does it preserve every exact event.outcomes label and avoid extra labels?
3. Priors: Are exact-match priors actually exact? Are related priors softened?
4. Base rate: Is elapsed time/current state accounted for?
5. Evidence hierarchy: official resolver/direct data > reputable current reporting > weak news.
6. Calibration: Is it overconfident, underconfident, or using the wrong question-type rule?
7. Retrieval honesty: Does it claim facts not present in the event or retrieved evidence?
8. Schema: Can the final adapter consume the probabilities exactly? Do not require
   probabilities to sum to 1 for Prophet Arena listed outcomes.

{CALIBRATION_RULE}

{FEW_SHOTS}

Return JSON only:
{{
  "verdict": "pass|revise|reject",
  "fatal_issues": ["issue"],
  "nonfatal_issues": ["issue"],
  "missing_information": ["information"],
  "corrections": ["specific correction"],
  "suggested_probabilities": {{
    "EXACT_OUTCOME_LABEL": 0.0
  }},
  "calibration_warning": "warning or null",
  "schema_warning": "warning or null",
  "summary": "short verifier judgment"
}}
""".strip()
    return [
        {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_synthesizer_messages(
    event: ProphetEvent,
    *,
    evidence: str,
    forecast_json: Any,
    verifier_json: Any,
) -> list[dict[str, str]]:
    payload = event.model_dump(mode="json")
    user = f"""
Produce the final Prophet Arena prediction after verifier review.

Event JSON:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Retrieved evidence:
{evidence}

Forecaster draft JSON(s):
{json.dumps(forecast_json, ensure_ascii=False, indent=2)}

Verifier JSON(s):
{json.dumps(verifier_json, ensure_ascii=False, indent=2)}

Synthesis rules:
- Compare the independent branch drafts and verifier critiques before deciding.
- If any verifier says revise or reject, incorporate the valid corrections.
- Preserve every exact event.outcomes label; do not add labels.
- Do not normalize, smear to uniform, or reject merely because listed probabilities do not
  sum to 1.
- Prefer official resolver/direct evidence over weak news.
- Keep noisy or weak-evidence categorical questions broad.
- Avoid extreme 0/1 probabilities unless the event is already mechanically settled.
- Return final JSON only.

Output JSON schema:
{{
  "probabilities": {{
    "EXACT_OUTCOME_LABEL": 0.0
  }},
  "rationale": "short explanation",
  "confidence": 0.0,
  "calibration_note": "short note"
}}
""".strip()
    return [
        {"role": "system", "content": SYNTHESIZER_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_messages(
    event: ProphetEvent,
    evidence: str = "No external retrieval evidence available.",
) -> list[dict[str, str]]:
    return build_forecaster_messages(event, evidence)
