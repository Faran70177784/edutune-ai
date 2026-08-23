"""Central configuration management for EduTune AI."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
EVALUATION_DATA_DIR = DATA_DIR / "evaluation"

CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
ADAPTERS_DIR = MODELS_DIR / "adapters"

LOGS_DIR = EXPERIMENTS_DIR / "logs"
WANDB_DIR = EXPERIMENTS_DIR / "wandb"


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

load_dotenv(PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

APP_NAME = os.getenv("APP_NAME", "EduTune AI")
APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DOMAIN = os.getenv("DOMAIN", "education")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

MODEL_ID = os.getenv(
    "MODEL_ID",
    "mistralai/Mistral-7B-Instruct-v0.3",
)

TRAINING_MODE = os.getenv("TRAINING_MODE", "qlora")


# ---------------------------------------------------------------------------
# Hugging Face
# ---------------------------------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN", "")


# ---------------------------------------------------------------------------
# Weights & Biases
# ---------------------------------------------------------------------------

WANDB_API_KEY = os.getenv("WANDB_API_KEY", "")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "edutune-ai")
WANDB_ENTITY = os.getenv("WANDB_ENTITY", "")


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Canonical random seed
SEED = int(os.getenv("SEED", "42"))

# Backward-compatible alias used by dataset/training modules
RANDOM_SEED = SEED


# ---------------------------------------------------------------------------
# Application metadata
# ---------------------------------------------------------------------------

APP_DESCRIPTION = (
    "EduTune AI is a domain-specific educational assistant "
    "fine-tuned using parameter-efficient techniques."
)


def get_project_paths() -> dict[str, Path]:
    """Return important project directories."""

    return {
        "project_root": PROJECT_ROOT,
        "config": CONFIG_DIR,
        "data": DATA_DIR,
        "raw_data": RAW_DATA_DIR,
        "processed_data": PROCESSED_DATA_DIR,
        "synthetic_data": SYNTHETIC_DATA_DIR,
        "evaluation_data": EVALUATION_DATA_DIR,
        "models": MODELS_DIR,
        "checkpoints": CHECKPOINTS_DIR,
        "adapters": ADAPTERS_DIR,
        "reports": REPORTS_DIR,
        "experiments": EXPERIMENTS_DIR,
        "logs": LOGS_DIR,
        "wandb": WANDB_DIR,
    }