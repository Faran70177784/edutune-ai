"""Tests for EduTune AI model comparison utilities."""

from evaluation.compare_models import (
    calculate_improvement,
    calculate_relative_improvement,
    compare_model_metrics,
)


def test_calculate_improvement():
    baseline = {
        "exact_match": 0.50,
        "token_overlap": 0.60,
    }

    finetuned = {
        "exact_match": 0.70,
        "token_overlap": 0.80,
    }

    result = calculate_improvement(
        baseline,
        finetuned,
    )

    assert result["exact_match"] == 0.20
    assert result["token_overlap"] == 0.20


def test_relative_improvement():
    baseline = {
        "exact_match": 0.50,
        "token_overlap": 0.50,
    }

    finetuned = {
        "exact_match": 0.75,
        "token_overlap": 0.60,
    }

    result = calculate_relative_improvement(
        baseline,
        finetuned,
    )

    assert result["exact_match"] == 50.0
    assert result["token_overlap"] == 20.0


def test_zero_baseline_relative_improvement():
    baseline = {
        "exact_match": 0.0,
        "token_overlap": 0.0,
    }

    finetuned = {
        "exact_match": 0.50,
        "token_overlap": 0.75,
    }

    result = calculate_relative_improvement(
        baseline,
        finetuned,
    )

    assert result["exact_match"] == 0.0
    assert result["token_overlap"] == 0.0


def test_compare_model_metrics():
    baseline = {
        "exact_match": 0.40,
        "token_overlap": 0.60,
    }

    finetuned = {
        "exact_match": 0.70,
        "token_overlap": 0.80,
    }

    result = compare_model_metrics(
        baseline,
        finetuned,
    )

    assert result["baseline"]["exact_match"] == 0.40
    assert result["finetuned"]["exact_match"] == 0.70
    assert result["absolute_improvement"]["exact_match"] == 0.30
    assert result["absolute_improvement"]["token_overlap"] == 0.20