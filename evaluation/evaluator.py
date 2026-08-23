"""Evaluation pipeline for EduTune AI."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import (
    aggregate_metrics,
    calculate_metrics,
)
from training.dataset_loader import load_training_datasets


def evaluate_predictions(
    predictions: list[str],
    references: list[str],
) -> dict[str, Any]:
    """Evaluate predictions against reference responses."""

    if len(predictions) != len(references):
        raise ValueError(
            "Predictions and references must contain "
            "the same number of records."
        )

    results = []

    for index, (prediction, reference) in enumerate(
        zip(predictions, references)
    ):
        metrics = calculate_metrics(
            prediction,
            reference,
        )

        results.append(
            {
                "index": index,
                "prediction": prediction,
                "reference": reference,
                **metrics,
            }
        )

    return {
        "metrics": aggregate_metrics(results),
        "results": results,
    }


def evaluate_test_dataset(
    predictions: list[str],
) -> dict[str, Any]:
    """Evaluate predictions against the project test dataset."""

    datasets = load_training_datasets()

    test_dataset = datasets["test"]

    references = [
        str(response)
        for response in test_dataset["response"]
    ]

    return evaluate_predictions(
        predictions,
        references,
    )


def save_evaluation_report(
    report: dict[str, Any],
    output_path: Path,
) -> None:
    """Save an evaluation report as JSON."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    datasets = load_training_datasets()

    test_dataset = datasets["test"]

    # Temporary baseline predictions.
    #
    # This is deliberately NOT presented as model output.
    # It only validates the evaluation pipeline before
    # the trained model is available.
    predictions = [
        str(response)
        for response in test_dataset["response"]
    ]

    report = evaluate_test_dataset(predictions)

    output_path = (
        PROJECT_ROOT
        / "data"
        / "evaluation"
        / "baseline_evaluation_report.json"
    )

    save_evaluation_report(
        report,
        output_path,
    )

    print("EduTune AI evaluation")
    print("--------------------------------")
    print(f"Test records: {len(test_dataset)}")
    print(
        "Exact match: "
        f"{report['metrics']['exact_match']:.4f}"
    )
    print(
        "Token overlap: "
        f"{report['metrics']['token_overlap']:.4f}"
    )
    print(f"Report: {output_path}")
    print("--------------------------------")
    print("Evaluation pipeline: PASSED")