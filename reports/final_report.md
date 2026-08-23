# EduTune AI — Final Project Report

**Project:** EduTune AI  
**Domain:** Education  
**Base Model:** `mistralai/Mistral-7B-Instruct-v0.3`  
**Fine-Tuning Approach:** QLoRA  
**Project Type:** Education-Focused Generative AI System  
**Development Environment:** Windows / Python 3.12.10  
**Testing Framework:** Pytest  
**Final Test Status:** 69/69 tests passed  
**CUDA:** Unavailable on development machine  
**Project Status:** Development and evaluation pipeline operational

---

## 1. Executive Summary

EduTune AI is an education-focused artificial intelligence project designed to support educational question answering, instructional assistance, model evaluation, dataset preparation, and future domain-specific fine-tuning.

The project is built around:

```text
mistralai/Mistral-7B-Instruct-v0.3

and is configured to use:

QLoRA

as the intended fine-tuning strategy.

The project implements a structured machine learning workflow covering:

Dataset construction
Dataset curation
Synthetic data generation
Dataset validation
Training dataset preparation
Tokenizer management
Hardware-aware model loading
Inference utilities
Evaluation metrics
Model comparison
Training configuration
Educational assistant functionality
Logging and utility functions
Automated testing
Evaluation reporting

The complete automated test suite currently passes:

69 passed

This confirms that the implemented application components and supporting pipelines satisfy the project's current automated test requirements.

The primary hardware limitation is that the development machine does not provide a CUDA-enabled NVIDIA GPU. Consequently, actual Mistral-7B model loading and GPU-based QLoRA training are safely blocked by the project's hardware-aware implementation.

This limitation does not prevent development, dataset preparation, pipeline validation, tokenizer testing, evaluation logic testing, or software-level verification.

2. Project Objectives

The main objectives of EduTune AI are to build a structured educational AI workflow capable of:

Preparing education-focused training data.
Curating and validating datasets.
Generating synthetic educational examples.
Preparing datasets for model training.
Supporting Mistral-7B as the configured base model.
Providing QLoRA-based fine-tuning configuration.
Detecting available hardware before expensive model operations.
Supporting educational inference workflows.
Evaluating model outputs using quantitative metrics.
Comparing baseline and fine-tuned model performance.
Maintaining reproducible experiments.
Providing automated software-level validation.
Producing structured technical reports.
3. System Overview

EduTune AI follows a modular architecture.

The major project components include:

EduTune AI
│
├── Configuration
│
├── Dataset Pipeline
│   ├── Dataset Building
│   ├── Dataset Curation
│   ├── Synthetic Generation
│   └── Dataset Validation
│
├── Training Pipeline
│   ├── QLoRA Configuration
│   ├── Training Configuration
│   ├── Hardware Detection
│   └── Hyperparameter Search
│
├── Model Layer
│   ├── Model Loading
│   ├── Tokenizer
│   └── Model Checkpoints
│
├── Inference Layer
│   ├── Educational Prompts
│   ├── Chat Prompts
│   └── Response Processing
│
├── Evaluation Layer
│   ├── Exact Match
│   ├── Token Overlap
│   ├── Metric Aggregation
│   └── Model Comparison
│
├── Assistant Layer
│   ├── Educational Requests
│   ├── Chat History
│   └── Response Formatting
│
├── Utilities
│   ├── Hardware Utilities
│   ├── Text Helpers
│   ├── Logging
│   └── Reproducibility
│
├── Reports
│
└── Automated Tests

This modular structure separates data processing, training, inference, evaluation, and application logic.

4. Technology Stack
4.1 Programming Language

The primary programming language is:

Python 3.12.10

Python is used throughout the project for:

Dataset processing
Model utilities
Training configuration
Evaluation
Inference
Testing
Application logic
4.2 Machine Learning Framework

The project uses:

PyTorch

PyTorch provides the underlying deep learning framework for model loading, hardware detection, and future training/inference workloads.

The current environment reports a CPU-only PyTorch installation.

4.3 Transformers

The project uses Hugging Face Transformers for:

Tokenizer loading
Model loading
Causal language model support
Mistral model integration

The configured model is:

mistralai/Mistral-7B-Instruct-v0.3
4.4 Fine-Tuning Strategy

The intended fine-tuning strategy is:

QLoRA

QLoRA is configured for memory-efficient fine-tuning of the selected large language model.

The project's training configuration specifies QLoRA as the fine-tuning mode.

4.5 Testing

Automated testing is implemented using:

pytest

The final test suite contains:

69 tests

with:

69 passed
0 failed
5. Dataset Pipeline

The project includes a complete dataset preparation workflow.

The dataset pipeline consists of:

Seed Dataset
      │
      ▼
Dataset Curation
      │
      ▼
Synthetic Data Generation
      │
      ▼
Dataset Validation
      │
      ▼
Training Dataset
      │
      ▼
Model Training

This structure makes the dataset workflow reproducible and modular.

6. Seed Dataset

The initial education seed dataset contains:

8 records

The dataset covers multiple educational categories.

Recorded categories include:

Biology
Computer Science
Economics
Mathematics
Physics

The seed dataset provides the initial examples used by the dataset curation and synthetic data generation stages.

7. Curated Dataset

The curated dataset contains:

8 records

The validation process reported:

Accepted records: 8
Rejected records: 0

The curated dataset therefore contains all eight seed records after the implemented curation stage.

The dataset validation process also checks for duplicate records and required fields.

8. Synthetic Dataset

Synthetic educational data was generated as part of the project workflow.

The generated synthetic dataset contains:

64 records

The synthetic data covers the following task types:

Concept explanation
Example generation
Question answering
Study guidance

The recorded distribution contains:

Concept explanation: 16
Example generation: 16
Question answering: 16
Study guidance: 16

The generated dataset provides additional educational examples for the future training pipeline.

9. Dataset Validation

Dataset validation is implemented as an independent pipeline stage.

The validation workflow checks dataset quality and produces a structured evaluation report.

The validation process records information including:

Record counts
Category distribution
Difficulty distribution
Task-type distribution
Duplicate detection
Dataset structure
Repeated phrase detection

The validation report is stored under:

data/evaluation/dataset_validation_report.json

A repeated phrase issue was detected in the synthetic dataset validation stage.

This demonstrates that the validation pipeline is capable of identifying potential quality issues rather than assuming that generated data is automatically perfect.

10. Model Configuration

The configured base model is:

mistralai/Mistral-7B-Instruct-v0.3

The project configuration identifies:

Domain:
education

Training mode:
qlora

The model is intended to serve as the foundation for future education-domain adaptation.

11. Hardware-Aware Model Loading

A major design feature of EduTune AI is hardware-aware model loading.

Before attempting to load the large language model, the project checks the available PyTorch device.

The current development environment reports:

Device: CPU
CUDA available: False
CUDA device count: 0
GPU name: None
GPU memory: None

Therefore:

Model loading: Blocked

This behavior is intentional.

The project does not attempt to load Mistral-7B weights when the required CUDA environment is unavailable.

12. Model Loading Safety

The model-loading implementation provides safe failure behavior.

When CUDA is unavailable, the project raises a controlled runtime condition instead of attempting unsupported model loading.

The recorded status is:

BLOCKED

with the reason:

CUDA is unavailable. Mistral-7B weights will not be loaded on this CPU-only machine.

This protects the development environment from unnecessary memory pressure and unsupported execution.

13. Tokenizer Implementation

EduTune AI includes a dedicated tokenizer utility.

The tokenizer is associated with:

mistralai/Mistral-7B-Instruct-v0.3

The tokenizer successfully loads in the current environment.

The recorded tokenizer information includes:

Vocabulary size: 32768
PAD token: </s>
PAD token ID: 2
EOS token: </s>
EOS token ID: 2
BOS token: <s>
BOS token ID: 1

The tokenizer also successfully encodes educational text during automated testing.

14. Inference Pipeline

The inference layer provides reusable functionality for educational prompts.

The pipeline supports:

Educational questions
Optional context
Chat messages
Prompt validation
Tokenization
Response generation interfaces
Hardware availability checks

The inference implementation also prevents empty prompts and invalid educational questions from proceeding through the pipeline.

15. Educational Assistant

The project includes an educational assistant layer designed to provide structured educational interactions.

The assistant supports:

Educational request validation
Educational prompt generation
Optional context
Response cleaning
Response formatting
Chat history
Chat reset functionality
Generator integration

The assistant architecture separates user interaction logic from model-generation logic.

This allows the underlying generator to be replaced or upgraded without rewriting the complete assistant layer.

16. Evaluation Framework

EduTune AI includes a dedicated evaluation framework.

The current evaluation metrics include:

Exact Match

Exact Match determines whether the normalized prediction exactly matches the reference.

Token Overlap

Token Overlap measures the overlap between tokens contained in the prediction and reference.

Metric Aggregation

The evaluation framework also supports aggregating individual evaluation results into summary metrics.

17. Baseline Evaluation

The configured baseline model is:

mistralai/Mistral-7B-Instruct-v0.3

However, the current development environment is CPU-only.

The baseline model report records:

Status: blocked
Device: cpu
CUDA available: false
CUDA device count: 0
GPU name: null

Therefore, an actual Mistral-7B GPU inference benchmark was not completed on the local development machine.

This distinction is important when interpreting the evaluation artifacts.

18. Baseline Evaluation Artifact

The project contains a baseline evaluation artifact:

data/evaluation/baseline_evaluation_report.json

The stored artifact reports:

Exact Match: 1.00
Token Overlap: 1.00

The artifact contains seven prediction/reference records.

Every stored prediction/reference pair has:

Exact Match: 1.00
Token Overlap: 1.00

These values demonstrate that the stored prediction strings exactly match the stored reference strings.

They should not be interpreted as successful Mistral-7B inference measurements because the baseline model was blocked by the CPU-only environment.

19. Benchmark Evaluation

The project includes a dedicated benchmark report:

reports/benchmark_report.md

The benchmark stage provides a structured basis for evaluating the project's evaluation pipeline and recorded model results.

Benchmark analysis is intended to support:

Baseline assessment
Fine-tuned model assessment
Metric comparison
Performance analysis
Future experiment tracking

The benchmark results should always be interpreted together with the model-loading and hardware status.

20. Model Comparison

EduTune AI includes model comparison utilities.

The comparison framework supports:

Metric comparison
Absolute improvement
Relative improvement
Zero-baseline handling

This allows future comparisons between:

Baseline Model
        vs.
Fine-Tuned Model

The comparison layer is designed to prevent invalid calculations when the baseline metric is zero.

21. Training Pipeline

The training pipeline is configured around:

QLoRA

The pipeline includes:

Training configuration loading
Hardware detection
Training environment summaries
Hyperparameter search space
Deterministic hyperparameter trials
CPU safety checks

The current environment prevents actual model training because CUDA is unavailable.

The training pipeline therefore performs hardware validation before model loading.

22. CPU Safety

The project explicitly prevents large-model training from proceeding in an unsupported CPU-only environment.

This behavior is verified by automated tests.

The test suite confirms that:

CPU environment blocks training before model loading

This is an important safety feature for local development.

23. Reproducibility

EduTune AI includes reproducibility utilities.

The project provides deterministic seed handling and supporting utilities to improve experimental consistency.

The automated test suite verifies that the seed utility returns the requested seed value.

Reproducibility is particularly important for:

Dataset generation
Synthetic data generation
Hyperparameter experiments
Evaluation
Model comparison
24. Utility Layer

The project includes reusable utility modules for:

Hardware detection
Text normalization
Text cleaning
Input validation
Safe filename creation
Logging
Random seed configuration
Project root resolution

These utilities reduce duplication across the project and provide centralized helper functionality.

25. Logging

A dedicated logging utility is included in the project.

The logger utility provides a consistent logging interface for project components.

This supports:

Debugging
Training monitoring
Evaluation logging
Error reporting
Application diagnostics
26. Automated Testing

Automated testing is one of the strongest completed components of the project.

The final test execution reports:

69 passed
14 warnings

No test failures were recorded.

The test suite covers:

Assistant functionality
Dataset validation
Dataset loading
Evaluation
Inference
Model comparison
Model loading
Tokenizer functionality
Training pipeline
Utilities
27. Test Coverage by Component

The test suite contains tests for the following major areas.

Assistant

Tests cover:

Educational request validation
Prompt construction
Context support
Response cleaning
Response extraction
Response formatting
Generator integration
Chat history
Chat reset
Dataset

Tests cover:

Dataset existence
Dataset records
Required fields
Instructions
Responses
Categories
Curated dataset
Duplicate detection
Dataset Loader

Tests cover:

Training dataset loading
Dataset sizes
Required training columns
Non-empty training data
Evaluation

Tests cover:

Text normalization
Exact match
Token overlap
Metric calculation
Metric aggregation
Inference

Tests cover:

Tokenizer loading
Encoding
Decoding
Educational prompts
Chat prompts
Empty input validation
Hardware readiness
Model Comparison

Tests cover:

Improvement calculation
Relative improvement
Zero baseline handling
Metric comparison
Models

Tests cover:

Device detection
Hardware summary
Model-loading availability
CPU safety
Tokenizer functionality
Special tokens
Text encoding
Training Pipeline

Tests cover:

Hardware summary
Training configuration
Environment summary
CPU safety
Hyperparameter search
Deterministic trials
Utilities

Tests cover:

Hardware summary
Device detection
CUDA availability
Text normalization
Text cleaning
Input validation
Safe filenames
Logging
Seed handling
Project root resolution
28. Final Test Result

The final project verification command was:

python -m pytest tests -v

The final result was:

====================== 69 passed, 14 warnings ======================

The project also successfully passed:

python -m compileall -q .

No compilation errors were reported.

29. Warning Status

The test execution reported PyTorch-related deprecation warnings associated with:

torch.jit.script_method

The warning recommends migration toward:

torch.compile

or:

torch.export

These warnings did not cause test failures and do not prevent the current project from operating.

They should be reviewed during future dependency or framework upgrades.

30. Current Hardware Environment

The current development environment reports:

Component	Status
Operating System	Windows
Python	3.12.10
Device	CPU
CUDA	Unavailable
CUDA Devices	0
GPU	None
Mistral-7B Loading	Blocked
QLoRA Training	Blocked
Tokenizer	Available
Software Tests	Passing

The project therefore distinguishes between software readiness and hardware readiness.

31. Project Strengths

The completed implementation demonstrates several important strengths.

31.1 Modular Architecture

The project separates:

Data
Training
Models
Inference
Evaluation
Assistant logic
Utilities
Reports

This improves maintainability.

31.2 Hardware Awareness

The system checks CUDA availability before loading or training the large model.

31.3 Safe Failure

Unsupported hardware conditions are handled explicitly instead of producing uncontrolled failures.

31.4 Dataset Quality Controls

The dataset pipeline includes curation, synthetic generation, and validation.

31.5 Automated Testing

The project has a substantial automated test suite with all 69 tests currently passing.

31.6 Reproducibility

Seed management and deterministic hyperparameter trials provide a foundation for repeatable experiments.

31.7 Evaluation Separation

The project distinguishes stored evaluation artifacts from actual model inference results.

This improves experimental integrity.

32. Current Limitations

Despite the completed software pipeline, several limitations remain.

32.1 No CUDA GPU

The current machine does not provide a CUDA-enabled NVIDIA GPU.

Therefore, the configured Mistral-7B model cannot currently be loaded using the intended GPU workflow.

32.2 No Local Mistral-7B Inference

Because model loading is blocked, the current environment does not provide a genuine local Mistral-7B inference benchmark.

32.3 No Local QLoRA Training

Actual QLoRA training requires compatible GPU hardware and sufficient memory.

The current CPU-only environment therefore blocks the training stage.

32.4 Synthetic Dataset Quality

Synthetic data requires continued quality review.

The dataset validation stage has already demonstrated the ability to identify repeated phrase issues.

32.5 Benchmark Interpretation

Stored evaluation metrics must not be confused with genuine model benchmark results when model inference was not executed.

33. Experimental Integrity

Maintaining a clear distinction between different types of results is a core requirement of this project.

EduTune AI separates:

Environment Validation

Determines whether the machine can perform the requested operation.

Model Evaluation

Measures the behavior of an actually loaded model.

Artifact Evaluation

Measures predictions and references already stored in an evaluation artifact.

These three stages must not be treated as interchangeable.

This distinction prevents unsupported claims about model performance.

34. Reproduction Requirements

A complete Mistral-7B baseline and QLoRA experiment should be performed in an environment that provides:

CUDA-compatible NVIDIA GPU
Compatible PyTorch installation
Compatible Transformers installation
Required QLoRA dependencies
Sufficient GPU memory
Access to the configured Hugging Face model
Valid model/tokenizer configuration

Once compatible hardware is available, the project can proceed to actual model loading, training, and inference.

35. Recommended Future Workflow

The recommended next workflow is:

1. Prepare CUDA-enabled environment
          │
          ▼
2. Verify GPU availability
          │
          ▼
3. Load Mistral-7B tokenizer
          │
          ▼
4. Load baseline model
          │
          ▼
5. Run baseline evaluation
          │
          ▼
6. Record baseline metrics
          │
          ▼
7. Execute QLoRA fine-tuning
          │
          ▼
8. Save adapter/checkpoint
          │
          ▼
9. Evaluate fine-tuned model
          │
          ▼
10. Compare baseline vs fine-tuned
          │
          ▼
11. Calculate improvement
          │
          ▼
12. Generate final experimental report
36. Future Fine-Tuned Model Evaluation

When compatible hardware becomes available, the fine-tuned model should be evaluated using the same evaluation methodology as the baseline.

The evaluation should record:

Model identifier
Checkpoint
Dataset version
Number of evaluation examples
Exact Match
Token Overlap
Additional evaluation metrics where appropriate
Hardware environment
Inference configuration

Using the same evaluation methodology makes the comparison more meaningful.

37. Baseline vs Fine-Tuned Comparison

The final comparison should follow the structure:

Metric	Baseline	Fine-Tuned	Improvement
Exact Match	Pending GPU evaluation	Pending QLoRA evaluation	Pending
Token Overlap	Pending GPU evaluation	Pending QLoRA evaluation	Pending

The current stored artifact contains:

Exact Match: 1.00
Token Overlap: 1.00

but these values represent stored prediction/reference comparisons rather than verified Mistral-7B inference on the current machine.

Therefore, they should not be used as the definitive baseline for a scientific model-performance claim.

38. Project Deliverables

The project currently contains the major components required for a structured EduTune AI implementation, including:

Configuration
Dataset pipeline
Synthetic data generation
Dataset validation
Training configuration
QLoRA configuration
Model utilities
Tokenizer utilities
Inference utilities
Educational assistant
Evaluation framework
Model comparison
Hardware detection
Logging
Reproducibility utilities
Automated tests
Baseline report
Benchmark report
Final report
39. Project Verification Checklist
Requirement	Status
Project structure	Completed
Configuration	Completed
Seed dataset	Completed
Dataset curation	Completed
Synthetic generation	Completed
Dataset validation	Completed
Training dataset preparation	Completed
Model configuration	Completed
Tokenizer integration	Completed
Hardware detection	Completed
Safe model loading	Completed
Inference utilities	Completed
Educational assistant	Completed
Evaluation metrics	Completed
Model comparison	Completed
Utility layer	Completed
Automated tests	Completed
Compilation check	Passed
Baseline report	Completed
Benchmark report	Completed
Fine-tuned evaluation	Pending compatible GPU
Actual QLoRA training	Pending compatible GPU
Final experimental comparison	Pending compatible GPU
40. Final Validation Summary

The project was validated using:

python -m pytest tests -v

Result:

69 passed

The project was also validated using:

python -m compileall -q .

Result:

Successful

Hardware readiness was verified using the model utilities.

Current status:

Device: CPU
CUDA available: False
CUDA devices: 0
Can load model: False

This confirms that the hardware-aware safeguards are functioning as intended.

41. Overall Project Status

The EduTune AI software and evaluation infrastructure is operational.

The following stages are complete:

Dataset Preparation
        ✓
Dataset Curation
        ✓
Synthetic Generation
        ✓
Dataset Validation
        ✓
Training Configuration
        ✓
Model Utilities
        ✓
Tokenizer
        ✓
Inference Utilities
        ✓
Assistant
        ✓
Evaluation
        ✓
Model Comparison
        ✓
Testing
        ✓
Reporting
        ✓

The following stages remain hardware-dependent:

Mistral-7B GPU Loading
        Pending

QLoRA Training
        Pending

Fine-Tuned Model Evaluation
        Pending

Verified Baseline-vs-Fine-Tuned Benchmark
        Pending
42. Conclusion

EduTune AI establishes a structured foundation for an education-focused generative AI system based on:

mistralai/Mistral-7B-Instruct-v0.3

with:

QLoRA

as the intended fine-tuning strategy.

The project successfully implements the supporting machine learning infrastructure required for dataset preparation, validation, model utilities, inference, evaluation, comparison, reproducibility, and automated testing.

The strongest current verification result is:

69 / 69 automated tests passed

and the project also successfully passes Python compilation checks.

The current development machine does not provide a CUDA-enabled NVIDIA GPU. The project's hardware-aware design correctly identifies this limitation and prevents unsupported Mistral-7B model loading and QLoRA training.

The stored baseline evaluation artifact reports perfect Exact Match and Token Overlap values, but these values represent stored prediction/reference comparisons and are not treated as verified Mistral-7B inference benchmarks.

Therefore, the project has reached a strong software-development and pipeline-validation stage, while the actual large-model training and GPU-based evaluation remain dependent on access to compatible CUDA hardware.

Once a suitable GPU environment is available, the existing project structure provides a clear path to:

Load the Mistral-7B baseline.
Execute genuine baseline inference.
Record reliable baseline metrics.
Perform QLoRA fine-tuning.
Evaluate the fine-tuned model.
Compare baseline and fine-tuned performance.
Quantify improvement.
Produce the final experimental benchmark.

EduTune AI therefore provides a reproducible and extensible foundation for continued development of an education-domain language model system.

43. Source Artifacts

Important project artifacts include:

config/settings.py
training_config.yaml

data/raw/education_seed.jsonl
data/processed/curated_dataset.jsonl
data/synthetic/synthetic_dataset.jsonl
data/evaluation/dataset_validation_report.json
data/evaluation/baseline_model_report.json
data/evaluation/baseline_evaluation_report.json

models/load_model.py
models/tokenizer.py

training/

inference/

evaluation/

assistant/

utils/

tests/

reports/baseline_report.md
reports/benchmark_report.md
reports/finetuned_report.md
reports/final_report.md