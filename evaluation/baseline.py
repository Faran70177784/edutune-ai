"""Baseline evaluation utilities for EduTune AI."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from config.settings import MODEL_ID  # noqa: E402
from evaluation.evaluator import evaluate_predictions  # noqa: E402
from models.load_model import load_model  # noqa: E402
from models.tokenizer import load_tokenizer  # noqa: E402
from training.dataset_loader import load_training_datasets  # noqa: E402


def check_baseline_hardware() -> dict[str, Any]:
    """Return hardware information for baseline evaluation."""

    cuda_available = torch.cuda.is_available()

    result: dict[str, Any] = {
        "device": "cuda" if cuda_available else "cpu",
        "cuda_available": cuda_available,
        "model_id": MODEL_ID,
        "status": "ready" if cuda_available else "blocked",
        "gpu_name": None,
        "device_count": 0,
    }

    if cuda_available:
        result["gpu_name"] = torch.cuda.get_device_name(0)
        result["device_count"] = torch.cuda.device_count()

    return result


def get_test_data() -> tuple[list[str], list[str]]:
    """Load test prompts and reference responses."""

    datasets = load_training_datasets()

    test_dataset = datasets["test"]

    prompts = [
        str(prompt)
        for prompt in test_dataset["prompt"]
    ]

    references = [
        str(response)
        for response in test_dataset["response"]
    ]

    return prompts, references


def generate_baseline_predictions(
    prompts: list[str],
    model: Any,
    tokenizer: Any,
    *,
    max_new_tokens: int = 128,
) -> list[str]:
    """Generate deterministic predictions from the base model."""

    if model is None:
        raise ValueError("Model cannot be None.")

    if tokenizer is None:
        raise ValueError("Tokenizer cannot be None.")

    if max_new_tokens <= 0:
        raise ValueError(
            "max_new_tokens must be greater than zero."
        )

    predictions: list[str] = []

    device = next(model.parameters()).device

    for prompt in prompts:
        prompt = str(prompt).strip()

        if not prompt:
            raise ValueError(
                "Test prompts cannot contain empty values."
            )

        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        prompt_length = encoded["input_ids"].shape[1]

        generated_tokens = generated[
            0,
            prompt_length:,
        ]

        prediction = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

        predictions.append(prediction)

    return predictions


def run_baseline_evaluation() -> dict[str, Any]:
    """
    Run base-model evaluation.

    CPU-only environments are deliberately blocked because
    the configured Mistral-7B workflow requires CUDA.
    """

    hardware = check_baseline_hardware()

    if not hardware["cuda_available"]:
        return {
            "status": "blocked",
            "reason": (
                "CUDA is unavailable. Baseline evaluation of "
                "Mistral-7B requires a CUDA-enabled GPU."
            ),
            "model_id": MODEL_ID,
            "hardware": hardware,
        }

    prompts, references = get_test_data()

    print("Loading tokenizer...")
    tokenizer = load_tokenizer()

    print("Loading base model...")
    model = load_model()

    print("Generating predictions...")

    predictions = generate_baseline_predictions(
        prompts,
        model,
        tokenizer,
    )

    evaluation = evaluate_predictions(
        predictions,
        references,
    )

    return {
        "status": "completed",
        "model_id": MODEL_ID,
        "hardware": hardware,
        "test_records": len(prompts),
        "metrics": evaluation["metrics"],
        "results": evaluation["results"],
    }


def save_baseline_report(
    report: dict[str, Any],
    output_path: Path | None = None,
) -> Path:
    """Save the baseline evaluation report."""

    if output_path is None:
        output_path = (
            PROJECT_ROOT
            / "data"
            / "evaluation"
            / "baseline_model_report.json"
        )

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

    return output_path


def main() -> int:
    """Run baseline evaluation from the command line."""

    print("EduTune AI baseline evaluation")
    print("=" * 50)

    hardware = check_baseline_hardware()

    print(f"Model: {MODEL_ID}")
    print(f"Device: {hardware['device']}")
    print(
        f"CUDA available: "
        f"{hardware['cuda_available']}"
    )

    if hardware["gpu_name"]:
        print(f"GPU: {hardware['gpu_name']}")

    print()

    report = run_baseline_evaluation()
    output_path = save_baseline_report(report)

    if report["status"] == "blocked":
        print("Baseline evaluation: BLOCKED")
        print("-" * 50)
        print(f"Reason: {report['reason']}")
        print(f"Report: {output_path}")
        return 0

    print("Baseline evaluation: COMPLETED")
    print("-" * 50)
    print(
        f"Test records: "
        f"{report['test_records']}"
    )
    print(
        f"Exact match: "
        f"{report['metrics']['exact_match']:.4f}"
    )
    print(
        f"Token overlap: "
        f"{report['metrics']['token_overlap']:.4f}"
    )
    print(f"Report: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())