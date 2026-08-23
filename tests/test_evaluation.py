"""Tests for EduTune AI evaluation utilities."""

from evaluation.metrics import (
    aggregate_metrics,
    calculate_metrics,
    exact_match_score,
    normalize_text,
    token_overlap_score,
)


def test_normalize_text():
    result = normalize_text(
        "  Hello,   WORLD!  "
    )

    assert result == "hello world"


def test_exact_match():
    assert (
        exact_match_score(
            "Hello World!",
            "hello world",
        )
        == 1.0
    )


def test_exact_match_difference():
    assert (
        exact_match_score(
            "Hello",
            "Goodbye",
        )
        == 0.0
    )


def test_token_overlap():
    score = token_overlap_score(
        "machine learning",
        "machine learning",
    )

    assert score == 1.0


def test_token_overlap_no_match():
    score = token_overlap_score(
        "physics",
        "database",
    )

    assert score == 0.0


def test_calculate_metrics():
    metrics = calculate_metrics(
        "machine learning",
        "machine learning",
    )

    assert "exact_match" in metrics
    assert "token_overlap" in metrics


def test_aggregate_metrics():
    results = [
        {
            "exact_match": 1.0,
            "token_overlap": 0.8,
        },
        {
            "exact_match": 0.0,
            "token_overlap": 0.6,
        },
    ]

    metrics = aggregate_metrics(results)

    assert metrics["exact_match"] == 0.5
    assert metrics["token_overlap"] == 0.7