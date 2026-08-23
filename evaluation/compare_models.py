"""Model comparison utilities for EduTune AI."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


METRIC_NAMES = (
    "exact_match",
    "token_overlap",
)


def _round_metric(value: float, digits: int = 10) -> float:
    """Round metric values to avoid floating-point artifacts."""

    return round(float(value), digits)


def calculate_improvement(
    baseline: dict[str, float],
    finetuned: dict[str, float],
) -> dict[str, float]:
    """Calculate absolute improvement from baseline to fine-tuned model."""

    return {
        metric: _round_metric(
            finetuned.get(metric, 0.0)
            - baseline.get(metric, 0.0)
        )
        for metric in METRIC_NAMES
    }


def calculate_relative_improvement(
    baseline: dict[str, float],
    finetuned: dict[str, float],
) -> dict[str, float]:
    """
    Calculate percentage improvement from baseline to fine-tuned model.

    A zero baseline produces 0.0 because relative improvement is
    undefined when the baseline metric is zero.
    """

    result: dict[str, float] = {}

    for metric in METRIC_NAMES:
        baseline_value = float(
            baseline.get(metric, 0.0)
        )
        finetuned_value = float(
            finetuned.get(metric, 0.0)
        )

        if baseline_value == 0.0:
            result[metric] = 0.0
            continue

        improvement = (
            (finetuned_value - baseline_value)
            / baseline_value
        ) * 100.0

        result[metric] = _round_metric(improvement)

    return result


def compare_model_metrics(
    baseline: dict[str, float],
    finetuned: dict[str, float],
) -> dict[str, Any]:
    """Create a complete comparison between two model metric sets."""

    absolute_improvement = calculate_improvement(
        baseline,
        finetuned,
    )

    relative_improvement = calculate_relative_improvement(
        baseline,
        finetuned,
    )

    return {
        "baseline": {
            metric: _round_metric(
                baseline.get(metric, 0.0)
            )
            for metric in METRIC_NAMES
        },
        "finetuned": {
            metric: _round_metric(
                finetuned.get(metric, 0.0)
            )
            for metric in METRIC_NAMES
        },
        "absolute_improvement": absolute_improvement,
        "relative_improvement_percent": relative_improvement,
    }


def save_comparison_report(
    report: dict[str, Any],
    output_path: Path,
) -> None:
    """Save the model comparison report as JSON."""

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


def build_demo_comparison() -> dict[str, Any]:
    """
    Build a demonstration comparison.

    These values are infrastructure-test values, not actual
    fine-tuned model results.
    """

    baseline = {
        "exact_match": 0.40,
        "token_overlap": 0.60,
    }

    finetuned = {
        "exact_match": 0.70,
        "token_overlap": 0.80,
    }

    return compare_model_metrics(
        baseline,
        finetuned,
    )


def main() -> int:
    """Run the model-comparison infrastructure demonstration."""

    report = build_demo_comparison()

    output_path = (
        PROJECT_ROOT
        / "data"
        / "evaluation"
        / "model_comparison_report.json"
    )

    save_comparison_report(
        report,
        output_path,
    )

    print("EduTune AI model comparison")
    print("--------------------------------")
    print("Comparison infrastructure: PASSED")
    print(
        "Exact-match improvement: "
        f"{report['absolute_improvement']['exact_match']:.2f}"
    )
    print(
        "Token-overlap improvement: "
        f"{report['absolute_improvement']['token_overlap']:.2f}"
    )
    print(
        "Exact-match relative improvement: "
        f"{report['relative_improvement_percent']['exact_match']:.2f}%"
    )
    print(
        "Token-overlap relative improvement: "
        f"{report['relative_improvement_percent']['token_overlap']:.2f}%"
    )
    print(f"Report: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())