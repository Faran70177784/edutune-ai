# EduTune AI — Baseline Evaluation Report

**Project:** EduTune AI  
**Domain:** Education  
**Base Model:** `mistralai/Mistral-7B-Instruct-v0.3`  
**Evaluation Type:** Baseline Model Evaluation  
**Training Approach:** QLoRA  
**Report Status:** Completed  
**Hardware Environment:** CPU-only  
**CUDA:** Unavailable  

---

## 1. Executive Summary

EduTune AI is an education-focused AI system designed to provide instructional assistance and support educational question answering.

This report documents the baseline evaluation stage of the project. The intended baseline model is:

`mistralai/Mistral-7B-Instruct-v0.3`

The project is configured for hardware-aware model evaluation. Because the current development environment does not provide a CUDA-enabled NVIDIA GPU, loading the 7B-parameter model for actual baseline inference was blocked.

The recorded hardware status confirms:

- Device: CPU
- CUDA available: `false`
- CUDA device count: `0`
- GPU name: `null`
- Baseline model loading: Blocked

Therefore, the recorded baseline model report should be interpreted as an **environment-readiness result**, not as a successful Mistral-7B inference benchmark.

A separate deterministic evaluation artifact contains reference/prediction comparisons with:

- Exact Match: `1.00`
- Token Overlap: `1.00`

These values represent the predictions stored in the evaluation artifact and should not be interpreted as evidence that the Mistral-7B model was successfully loaded or evaluated on the current CPU-only machine.

---

## 2. Baseline Model

### 2.1 Model Identity

The configured baseline model is:

```text
mistralai/Mistral-7B-Instruct-v0.3

This model is used as the base model for the EduTune AI fine-tuning and evaluation workflow.

2.2 Intended Role

The baseline model provides the reference point against which future EduTune AI fine-tuning and model improvements can be evaluated.

The evaluation workflow is intended to support comparison between:

Base/baseline model behavior
Fine-tuned model behavior
Model performance across defined educational evaluation samples
3. Evaluation Environment

The baseline evaluation was performed in a CPU-only development environment.

3.1 Hardware Status
Component	Status
Device	CPU
CUDA Available	No
CUDA Device Count	0
GPU Name	None
GPU Memory	Not available
Baseline Model Loading	Blocked

The baseline model report explicitly records that CUDA is unavailable and that Mistral-7B baseline evaluation requires a CUDA-enabled GPU.

3.2 Evaluation Constraint

The current environment cannot load the configured Mistral-7B model using the project's intended 4-bit QLoRA-compatible workflow.

The evaluation was therefore designed to fail safely rather than attempting to load large model weights on an unsupported CPU-only environment.

4. Baseline Evaluation Status

The recorded baseline model status is:

status: blocked

The recorded reason is:

CUDA is unavailable. Baseline evaluation of Mistral-7B requires a CUDA-enabled GPU.

This is an expected hardware limitation rather than an application failure.

The hardware-aware implementation prevents the system from proceeding with model loading when CUDA is unavailable.

4.1 Model Loading Result
Model ID:
mistralai/Mistral-7B-Instruct-v0.3

Device:
cpu

CUDA available:
false

Device count:
0

GPU:
none

Status:
blocked
5. Evaluation Dataset

The baseline evaluation artifact contains prediction/reference comparisons indexed from 0 through 6.

The recorded examples cover educational concepts from multiple subject areas, including:

Computer Science
Biology
Mathematics
Physics

Examples represented in the evaluation artifact include:

Difference between a variable and a constant
Role of mitochondria in a cell
Photosynthesis
Pythagorean theorem
Newton's second law

The evaluation artifact stores both a prediction and a corresponding reference for each evaluation item.

6. Evaluation Metrics

The evaluation artifact records two primary metrics.

6.1 Exact Match

Exact Match measures whether the normalized prediction exactly matches the reference answer.

Recorded value:

1.00
6.2 Token Overlap

Token Overlap measures the degree of overlap between the prediction and reference text.

Recorded value:

1.00
6.3 Recorded Metrics
Metric	Score
Exact Match	1.00
Token Overlap	1.00
7. Interpretation of Recorded Metrics

The evaluation artifact reports perfect scores for the stored prediction/reference comparisons.

However, these results must be interpreted carefully.

The baseline model report independently confirms that:

CUDA available: false

and:

status: blocked

Therefore, the 1.00 metric values should not be described as successful Mistral-7B benchmark results from the current CPU environment.

Instead, they demonstrate that the stored prediction strings in the evaluation artifact exactly correspond to their stored reference strings.

This distinction is important for maintaining evaluation integrity.

8. Evaluation Results
8.1 Summary
Evaluation Component	Result
Target Model	Mistral-7B-Instruct-v0.3
Model Loading	Blocked
Evaluation Hardware	CPU
CUDA	Unavailable
Exact Match	1.00
Token Overlap	1.00
Successful GPU Model Inference	No
8.2 Item-Level Results

The evaluation artifact contains seven indexed prediction/reference records.

Index	Exact Match	Token Overlap
0	1.00	1.00
1	1.00	1.00
2	1.00	1.00
3	1.00	1.00
4	1.00	1.00
5	1.00	1.00
6	1.00	1.00

Every stored prediction/reference pair in the artifact has matching metric values of 1.00.

9. Qualitative Analysis

The stored evaluation examples demonstrate that the evaluation pipeline is capable of producing structured prediction/reference comparisons.

The examples contain educational responses addressing foundational concepts and explaining their relevance to students.

The responses generally follow an instructional format, including concepts such as:

Definitions
Main ideas
Foundational understanding
Practical applications
Connections to related topics

This is aligned with the intended educational domain of EduTune AI.

However, because the actual Mistral-7B model was not loaded in the current environment, this report does not claim that these characteristics represent measured behavior of the deployed Mistral-7B model.

10. Baseline Strengths

The baseline evaluation workflow demonstrates several strengths.

10.1 Hardware Awareness

The project detects the available device before attempting model loading.

This prevents unsupported CPU environments from attempting to load the large baseline model.

10.2 Safe Failure Behavior

Instead of producing an uncontrolled memory or runtime failure, the system reports that CUDA is unavailable and blocks model loading.

10.3 Reproducible Evaluation Structure

The evaluation artifact stores:

Evaluation index
Prediction
Reference
Exact Match
Token Overlap

This provides a structured foundation for later model comparisons.

10.4 Clear Evaluation Metrics

The project has defined quantitative metrics that can be reused when comparing baseline and fine-tuned models.

11. Limitations

The primary limitation of this baseline evaluation is hardware availability.

11.1 No CUDA GPU

The current environment has:

CUDA available: false
Device count: 0

Consequently, the configured Mistral-7B model cannot be loaded using the intended GPU-based workflow.

11.2 No Actual Baseline Model Inference

Because model loading was blocked, this evaluation does not provide a genuine end-to-end Mistral-7B inference benchmark on the current machine.

11.3 Metric Interpretation

The perfect metric values in the evaluation artifact should not be treated as evidence of model quality.

They reflect the stored prediction/reference comparisons and must be kept separate from successful model inference results.

11.4 Hardware-Dependent Evaluation

A complete baseline model evaluation requires an environment capable of loading the configured Mistral-7B model.

12. Reproducibility

The baseline evaluation can be reproduced in a compatible CUDA-enabled environment using the project's existing evaluation pipeline.

Before running model inference, the environment should provide:

CUDA-compatible hardware
A compatible PyTorch installation
Required Transformers dependencies
Access to the configured Hugging Face model
Sufficient GPU memory for the selected quantization/loading configuration

The project should continue using hardware checks before model loading.

13. Recommended Baseline Evaluation Procedure

For a complete baseline evaluation on compatible hardware:

Verify CUDA availability.
Verify the target GPU.
Load the configured tokenizer.
Load mistralai/Mistral-7B-Instruct-v0.3.
Run the baseline evaluation dataset.
Generate predictions.
Calculate Exact Match.
Calculate Token Overlap.
Record inference/evaluation results.
Save the baseline report.
Use the results as the reference point for fine-tuned model evaluation.
14. Baseline Evaluation Integrity

To maintain reliable experimental results, future reports should clearly distinguish between:

Environment Validation

Determines whether the required hardware and dependencies are available.

Model Evaluation

Measures the actual behavior of the loaded model against an evaluation dataset.

Stored Artifact Evaluation

Measures the contents of an already-generated prediction/reference artifact.

These are separate evaluation stages and should not be presented as equivalent.

15. Conclusion

The EduTune AI baseline evaluation pipeline is operational and correctly detects the limitations of the current CPU-only environment.

The configured baseline model is:

mistralai/Mistral-7B-Instruct-v0.3

The current environment reports:

Device: CPU
CUDA: unavailable
GPU devices: 0

As a result, actual Mistral-7B baseline model loading is blocked.

The stored evaluation artifact reports:

Exact Match: 1.00
Token Overlap: 1.00

These results confirm that the stored prediction/reference pairs match exactly, but they should not be interpreted as successful Mistral-7B inference results from the current CPU-only environment.

The baseline stage therefore establishes a clear and reproducible foundation for the next stages of the EduTune AI project, particularly benchmark analysis and fine-tuned model comparison.

16. Next Steps

The next evaluation stages should focus on:

Benchmarking the evaluation pipeline.
Evaluating the fine-tuned model when compatible hardware is available.
Comparing baseline and fine-tuned model metrics.
Measuring relative improvement.
Documenting model limitations and hardware requirements.
Producing the final model comparison report.
17. Source Artifacts

The baseline evaluation is based on the following project artifacts:

data/evaluation/baseline_model_report.json
data/evaluation/baseline_evaluation_report.json

The baseline model report records the hardware-aware model loading status.

The baseline evaluation report contains the stored prediction/reference comparisons and their calculated metrics.