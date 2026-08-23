"""Evaluation metrics for EduTune AI."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""

    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)

    return text.strip()


def exact_match_score(
    prediction: str,
    reference: str,
) -> float:
    """Calculate exact-match score."""

    return float(
        normalize_text(prediction)
        == normalize_text(reference)
    )


def token_overlap_score(
    prediction: str,
    reference: str,
) -> float:
    """Calculate token-level F1 overlap."""

    prediction_tokens = normalize_text(prediction).split()
    reference_tokens = normalize_text(reference).split()

    if not prediction_tokens or not reference_tokens:
        return 0.0

    prediction_counts = Counter(prediction_tokens)
    reference_counts = Counter(reference_tokens)

    common = prediction_counts & reference_counts
    overlap = sum(common.values())

    if overlap == 0:
        return 0.0

    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def calculate_metrics(
    prediction: str,
    reference: str,
) -> dict[str, float]:
    """Calculate the core EduTune AI evaluation metrics."""

    return {
        "exact_match": exact_match_score(
            prediction,
            reference,
        ),
        "token_overlap": token_overlap_score(
            prediction,
            reference,
        ),
    }


def aggregate_metrics(
    results: list[dict[str, Any]],
) -> dict[str, float]:
    """Average metric values across evaluation examples."""

    if not results:
        return {
            "exact_match": 0.0,
            "token_overlap": 0.0,
        }

    metric_names = [
        "exact_match",
        "token_overlap",
    ]

    return {
        metric: sum(
            float(result.get(metric, 0.0))
            for result in results
        ) / len(results)
        for metric in metric_names
    }


if __name__ == "__main__":
    prediction = (
        "Photosynthesis allows plants to convert "
        "light energy into chemical energy."
    )

    reference = (
        "Photosynthesis is the process by which "
        "plants convert light energy into chemical energy."
    )

    metrics = calculate_metrics(
        prediction,
        reference,
    )

    print("EduTune AI evaluation metrics")
    print("--------------------------------")
    print(f"Exact match: {metrics['exact_match']:.4f}")
    print(f"Token overlap: {metrics['token_overlap']:.4f}")
    print("--------------------------------")
    print("Metrics: PASSED")