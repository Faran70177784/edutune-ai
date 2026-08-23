"""Baseline benchmarking for EduTune AI."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import MODEL_ID
from evaluation.evaluator import evaluate_predictions
from models.load_model import load_model
from models.tokenizer import load_tokenizer
from training.dataset_loader import load_training_datasets


def check_benchmark_hardware() -> dict[str, Any]:
    """Check whether the environment can run the baseline model."""

    cuda_available = torch.cuda.is_available()

    result = {
        "cuda_available": cuda_available,
        "device": "cuda" if cuda_available else "cpu",
        "model_id": MODEL_ID,
    }

    if cuda_available:
        result["status"] = "ready"
        result["device_name"] = torch.cuda.get_device_name(0)
    else:
        result["status"] = "blocked"
        result["device_name"] = None

    return result


def build_test_prompts() -> list[str]:
    """Return prompts from the project test dataset."""

    datasets = load_training_datasets()

    return [
        str(prompt)
        for prompt in datasets["test"]["prompt"]
    ]


def get_test_references() -> list[str]:
    """Return reference responses from the test dataset."""

    datasets = load_training_datasets()

    return [
        str(response)
        for response in datasets["test"]["response"]
    ]


def generate_predictions(
    prompts: list[str],
    model,
    tokenizer,
    max_new_tokens: int = 128,
) -> list[str]:
    """Generate model predictions for test prompts."""

    predictions = []

    device = next(model.parameters()).device

    for prompt in prompts:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][
            inputs["input_ids"].shape[1]:
        ]

        prediction = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

        predictions.append(prediction)

    return predictions


def run_baseline_benchmark() -> dict[str, Any]:
    """Run the base-model benchmark."""

    hardware = check_benchmark_hardware()

    if not hardware["cuda_available"]:
        return {
            "status": "blocked",
            "reason": (
                "CUDA is unavailable. The configured "
                "Mistral-7B baseline benchmark requires "
                "a CUDA-enabled environment."
            ),
            "hardware": hardware,
            "model_id": MODEL_ID,
        }

    datasets = load_training_datasets()

    prompts = [
        str(prompt)
        for prompt in datasets["test"]["prompt"]
    ]

    references = [
        str(response)
        for response in datasets["test"]["response"]
    ]

    print("Loading tokenizer...")
    tokenizer = load_tokenizer()

    print("Loading base model...")
    model = load_model()

    print("Generating baseline predictions...")

    predictions = generate_predictions(
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


def save_benchmark_report(
    report: dict[str, Any],
) -> Path:
    """Save the benchmark report."""

    output_path = (
        PROJECT_ROOT
        / "data"
        / "evaluation"
        / "baseline_benchmark_report.json"
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
    """Run the baseline benchmark."""

    print("EduTune AI baseline benchmark")
    print("=" * 50)
    print(f"Model: {MODEL_ID}")

    hardware = check_benchmark_hardware()

    print()
    print("Hardware")
    print("-" * 50)
    print(f"Device: {hardware['device']}")
    print(
        f"CUDA available: "
        f"{hardware['cuda_available']}"
    )

    if hardware["device_name"]:
        print(
            f"GPU: "
            f"{hardware['device_name']}"
        )

    print()

    report = run_baseline_benchmark()

    output_path = save_benchmark_report(report)

    if report["status"] == "blocked":
        print("Baseline benchmark: BLOCKED")
        print("-" * 50)
        print(f"Reason: {report['reason']}")
        print()
        print(f"Report: {output_path}")
        return 0

    print("Baseline benchmark: COMPLETED")
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
    print()
    print(f"Report: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())