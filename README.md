# EduTune AI

## Domain-Specific Educational AI Platform

EduTune AI is an enterprise-oriented educational AI platform designed to
provide clear, accurate, and student-friendly assistance within the education
domain.

The project demonstrates an end-to-end AI development workflow covering
dataset engineering, synthetic data generation, validation, evaluation,
hardware-aware model loading, inference, and baseline-versus-fine-tuned model
comparison.

---

## Project Overview

EduTune AI is built around the Mistral-7B-Instruct-v0.3 foundation model and
a parameter-efficient QLoRA fine-tuning workflow.

The platform is designed to support:

- Educational question answering
- Subject-aware responses
- Difficulty-aware educational assistance
- Structured prompt engineering
- Synthetic educational dataset generation
- Dataset curation and validation
- Baseline model evaluation
- Fine-tuned model comparison
- Hardware-aware inference
- Safe CPU-only execution
- Streamlit-based enterprise dashboard

The current development environment is CPU-only. Therefore, large Mistral-7B
model weights are intentionally blocked from loading when CUDA is unavailable.

---

## Key Features

### Dataset Engineering

- Seed educational dataset creation
- Dataset curation
- Synthetic data generation
- Dataset validation
- Duplicate detection
- Category and difficulty analysis
- Training/validation/test dataset preparation

### Model Pipeline

- Mistral-7B-Instruct-v0.3
- QLoRA fine-tuning architecture
- Hardware-aware model loading
- CUDA availability detection
- Safe model initialization
- Tokenizer management

### Inference

- Educational prompt templates
- Subject-aware prompts
- Difficulty-aware prompts
- Educational chat prompts
- Controlled text generation
- Safe generation error handling
- CPU-only inference protection

### Evaluation

- Exact-match evaluation
- Token-overlap F1 evaluation
- Aggregate evaluation metrics
- Baseline evaluation infrastructure
- Model comparison infrastructure
- Absolute improvement analysis
- Relative improvement analysis

### Application

- Enterprise-style Streamlit interface
- Dashboard
- Educational Assistant
- Evaluation
- System Information
- Runtime hardware diagnostics
- Model and training configuration visibility

---

## Technology Stack

| Category | Technology |
|---|---|
| Programming Language | Python 3.12 |
| Application Framework | Streamlit |
| Foundation Model | Mistral-7B-Instruct-v0.3 |
| Fine-Tuning | QLoRA |
| NLP Framework | Hugging Face Transformers |
| Deep Learning | PyTorch |
| Dataset Processing | Hugging Face Datasets |
| Configuration | YAML / python-dotenv |
| Testing | pytest |
| Version Control | Git / GitHub |
| Experiment Tracking | Weights & Biases integration |
| Interface | Streamlit |

---

## Architecture

```text
                         EduTune AI
                              |
             +----------------+----------------+
             |                                 |
       Data Engineering                   Application
             |                                 |
     +-------+-------+               +---------+---------+
     |       |       |               |         |         |
   Seed   Curate  Synthetic       Dashboard  Assistant  System
     |       |       |               |         |         |
     +-------+-------+               +---------+---------+
             |
         Validation
             |
       Train / Eval Data
             |
      +------+------+
      |             |
   Training      Evaluation
      |             |
    QLoRA       Baseline /
      |         Comparison
      |             |
      +------+------+
             |
       Mistral-7B
             |
       Inference Layer
             |
      Hardware Safety
             |
        CUDA / GPU