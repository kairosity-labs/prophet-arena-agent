from __future__ import annotations

from prophet_arena_agent.forecast import prediction_from_model_json, uniform_prediction
from prophet_arena_agent.models import ProphetEvent


def test_uniform_prediction_matches_all_outcomes() -> None:
    event = ProphetEvent(title="Who wins?", outcomes=["A", "B", "C"])

    prediction = uniform_prediction(event)

    assert [item.market for item in prediction.probabilities] == ["A", "B", "C"]
    assert all(item.probability == 1 / 3 for item in prediction.probabilities)


def test_prediction_preserves_exact_labels() -> None:
    event = ProphetEvent(title="Who wins?", outcomes=["Pittsburgh", "Atlanta"])

    prediction = prediction_from_model_json(
        event,
        {"probabilities": {"Pittsburgh": 0.68, "Atlanta": 0.32}},
    )

    assert prediction.model_dump(mode="json") == {
        "probabilities": [
            {"market": "Pittsburgh", "probability": 0.68},
            {"market": "Atlanta", "probability": 0.32},
        ]
    }


def test_binary_p_yes_maps_to_yes_no() -> None:
    event = ProphetEvent(title="Will it happen?", outcomes=["Yes", "No"])

    prediction = prediction_from_model_json(event, {"p_yes": 0.72})

    assert prediction.probabilities[0].market == "Yes"
    assert prediction.probabilities[0].probability == 0.72
    assert prediction.probabilities[1].market == "No"
    assert prediction.probabilities[1].probability == 0.28


def test_prediction_does_not_normalize_or_replace_all_zero_outputs() -> None:
    event = ProphetEvent(title="Which listed teams can win?", outcomes=["A", "B", "C"])

    prediction = prediction_from_model_json(
        event,
        {"probabilities": {"A": 0.0, "B": 0.0, "C": 0.0}},
    )

    assert prediction.model_dump(mode="json") == {
        "probabilities": [
            {"market": "A", "probability": 0.0},
            {"market": "B", "probability": 0.0},
            {"market": "C", "probability": 0.0},
        ]
    }


def test_prediction_preserves_non_sum_to_one_probabilities() -> None:
    event = ProphetEvent(title="Which listed teams can win?", outcomes=["A", "B", "C"])

    prediction = prediction_from_model_json(
        event,
        {"probabilities": {"A": 0.02, "B": 0.0, "C": 0.4}},
    )

    assert [item.probability for item in prediction.probabilities] == [0.02, 0.0, 0.4]
