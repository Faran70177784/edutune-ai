"""Hyperparameter-search utilities for EduTune AI."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.model_setup import load_training_config  # noqa: E402


def build_search_space(config: dict[str, Any] | None = None) -> dict[str, list[Any]]:
    """Return the configured hyperparameter search space."""
    config = config or load_training_config()
    search = config.get("hyperparameter_search", {})

    return {
        "learning_rate": list(search.get("learning_rates", [])),
        "lora_rank": list(search.get("lora_ranks", [])),
        "lora_dropout": list(search.get("dropout_values", [])),
    }


def generate_trials(config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Generate deterministic trial combinations from the configured space."""
    space = build_search_space(config)

    if not all(space.values()):
        return []

    keys = tuple(space)
    return [
        dict(zip(keys, values))
        for values in itertools.product(*(space[key] for key in keys))
    ]


def save_search_plan(
    trials: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Save a search plan without launching expensive training runs."""
    path = Path(output_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"status": "planned", "trial_count": len(trials), "trials": trials},
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def main() -> int:
    """Print the configured hyperparameter search plan."""
    config = load_training_config()
    enabled = bool(config.get("hyperparameter_search", {}).get("enabled", False))
    trials = generate_trials(config)

    print("EduTune AI hyperparameter search")
    print("--------------------------------")
    print(f"Enabled: {enabled}")
    print(f"Candidate trials: {len(trials)}")

    if not enabled:
        print("Search execution: BLOCKED")
        print("Reason: hyperparameter_search.enabled is false.")
        return 0

    for index, trial in enumerate(trials, start=1):
        print(f"Trial {index}: {trial}")

    output = save_search_plan(trials, "data/evaluation/hyperparameter_search_plan.json")
    print(f"Plan: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
