# PromptOps Multi-Model Benchmarking Protocol

## 1. Executive Summary

This document specifies the official, reproducible benchmarking protocol for evaluating Large Language Models (LLMs) on GitHub issue classification and triage within the PromptOps platform.

The protocol establishes standardized rules for running, recording, and comparing local Ollama models on the **Expanded Golden Dataset V2** (`datasets/v2/github_issues.json`, 50 cases). By controlling prompt templates, inference parameters, warm-up conditions, and execution iterations, this protocol ensures fair, scientifically defensible model evaluations.

---

## 2. Models Under Benchmark

The target models evaluated under this protocol are:

| Model Identifier | Ollama Tag | Architecture / Parameters | Context Window | Default Quantization |
| :--- | :--- | :--- | :--- | :--- |
| **Gemma 3** | `gemma3:4b` | Google Gemma 3 (~4 Billion parameters) | 8,192 tokens | Q4_K_M |
| **Mistral** | `mistral:latest` | Mistral AI (~7 Billion parameters) | 32,768 tokens | Q4_0 |
| **Llama 3** | `llama3:latest` | Meta Llama 3 (~8 Billion parameters) | 8,192 tokens | Q4_0 |

### Model-Specific Considerations:
- **Quantization:** Standard Ollama library pulls utilize default 4-bit quantization (`Q4_K_M` or `Q4_0`). All comparisons assume standard library tags.
- **System Prompt Handling:** Certain models (e.g., Llama 3) natively expect system prompts wrapped within specific role delimiters (`<|start_header_id|>system<|end_header_id|>`), whereas Gemma and Mistral handle direct system instructions. For standard benchmark parity, all models are invoked via the identical Ollama API interface and prompt template (`prompt/v1.yaml`).

---

## 3. Benchmarking Environment Controls

To guarantee comparability across runs and environments, the following environmental parameters must be captured and logged for every benchmark execution:

1. **Hardware Configuration:**
   - Host CPU architecture and core count.
   - Host GPU / VRAM availability (or CPU inference mode).
   - Available host RAM and background resource utilization.
2. **Software Environment:**
   - Operating System and build version.
   - Python runtime version (`3.12.x`).
   - Ollama server version (e.g. `ollama --version`).
   - Repository Git commit hash.
3. **Artifact Hashes:**
   - SHA-256 hash of `datasets/v2/github_issues.json`.
   - SHA-256 hash of `prompt/v1.yaml`.
   - SQLite database path (`data/evaluation.db` or isolated validation DB).

---

## 4. Benchmark Execution Protocol

To isolate transient system overhead and cold-start model loading delays from core inference metrics, benchmark runs must follow a strict **Warm-Up + Repeated Execution** procedure.

### A. Phase 1: Warm-Up (Excluded from Metrics)
1. Before recording metrics for a given model, execute **exactly one warm-up inference** using a dedicated non-benchmark test case or synthetic sample.
2. **Purpose:** Forces Ollama to load model weights from storage into VRAM/system memory, establish internal socket pools, and compile compute graphs.
3. **Exclusion:** The latency, parsing status, and outputs of the warm-up inference **must be discarded** and never mixed into benchmark aggregates.

### B. Phase 2: Repeated Measured Executions ($N \ge 3$)
1. Each model must evaluate the complete 50-case Golden Dataset V2 for a minimum of **three (3) independent repetitions** ($R_1, R_2, R_3$).
2. **Inference Parameters (Held Constant):**
   - `temperature`: `0.0` (greedy deterministic decoding)
   - `max_tokens`: `150`
   - Prompt Version: `v1` (`prompt/v1.yaml`)
   - Output Schema: JSON object `{category, priority, summary}`
3. **Concurrency:** Single-threaded or controlled concurrency (concurrency limit = 5 via Semaphore) to prevent thread starvation or socket pool saturation.
4. **Inter-Run Rest Period:** A 10-second idle period between repetition runs to allow thermal throttling or garbage collection to stabilize.

---

## 5. Metrics to Collect and Analyze

Every benchmark execution produces a structured `EvaluationResult` capturing three core dimensions: **Classification Quality**, **Performance / Latency**, and **Reliability**.

### A. Classification Quality Metrics
1. **Overall / Composite Accuracy:**
   $$\text{Composite Accuracy} = \frac{\sum \mathbb{I}(\text{Category Correct} \land \text{Priority Correct} \land \text{Summary Valid})}{N} \times 100$$
2. **Category Accuracy:**
   $$\text{Category Accuracy} = \frac{\sum \mathbb{I}(\text{Normalized Category Correct})}{N} \times 100$$
3. **Priority Accuracy:**
   $$\text{Priority Accuracy} = \frac{\sum \mathbb{I}(\text{Normalized Priority Correct})}{N} \times 100$$
4. **Per-Class Metrics (Category & Priority):**
   - Precision: $P_c = \frac{TP_c}{TP_c + FP_c}$
   - Recall: $R_c = \frac{TP_c}{TP_c + FN_c}$
   - F1 Score: $F1_c = \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$
   - Support: Number of true ground-truth instances for class $c$.
5. **Macro and Weighted Averages:**
   - Macro F1: Unweighted arithmetic mean across all target classes.
   - Weighted F1: Support-weighted mean across target classes.
6. **2D Confusion Matrices:**
   Full $8 \times 9$ matrix for categories (including `<error>` column for unparseable predictions) and $3 \times 4$ matrix for priorities.

### B. Performance & Latency Metrics
Calculated across all valid inference requests in each repetition:
1. **Average Latency:** Mean milliseconds per case ($\bar{L}$).
2. **Median Latency ($P_{50}$):** 50th percentile latency.
3. **Minimum Latency ($L_{\min}$):** Fastest recorded inference.
4. **Maximum Latency ($L_{\max}$):** Slowest recorded inference.
5. **Standard Deviation ($\sigma_L$):** Latency variance across cases.

### C. Reliability & Robustness Metrics
1. **Inference Success Rate:** Percentage of cases completing without socket timeout or engine exception.
2. **JSON Schema Compliance Rate:** Percentage of responses returning valid, parseable JSON conforming to the output schema.
3. **Error Retention:** All raw LLM responses and tracebacks preserved in `CaseResult.raw_response` and `CaseResult.error`.
4. **Repeatability (Variance across Repetitions):**
   - Standard deviation of composite accuracy across repetitions ($\sigma_{\text{acc}}$).
   - Standard deviation of macro F1 across repetitions ($\sigma_{\text{F1}}$).
   - Delta between minimum and maximum repetition accuracy.

---

## 6. Benchmark Fairness & Scientific Integrity

To maintain scientific rigor and prevent misleading claims, all PromptOps benchmarks must strictly adhere to the following principles:

### 1. Controlled Comparisons
- **Model Comparisons:** When comparing Model A vs Model B, hold the **dataset** (V2, 50 cases) and **prompt** (`v1`) strictly constant.
- **Prompt Comparisons:** When comparing Prompt v1 vs Prompt v2, hold the **model** (e.g. `gemma3:4b`) strictly constant.
- **No Conflation:** Cross-model performance variations **must never be labeled as prompt regressions**. They are model architectural differences.

### 2. Fair Handling of Missing and Error Outputs
- Incomplete, missing, or malformed JSON responses **must never be silently dropped or omitted** from the denominator.
- Errors are explicitly classified as failed cases and recorded as False Negatives in precision/recall calculations.
- Missing metrics in historical runs must display as `N/A`, never coerced to `0.0`.

### 3. Latency Fairness
- Latency comparisons must exclude model-loading cold starts via the mandatory warm-up step.
- Latency benchmarks must run under equivalent host load conditions without concurrent CPU/GPU intensive background tasks.

### 4. Sample Size & Statistical Significance
- Results from the 5-case V1 dataset must **never** be cited as statistically representative or conclusive.
- Results from the 50-case V2 dataset represent a solid benchmark, but claims of statistical significance (e.g., "Model A is definitively superior to Model B") must report confidence intervals (95% CI) and repetition variance rather than raw point estimates.

---

## 7. Reporting & Artifact Retention

Every official benchmark run conducted under this protocol must output the following artifacts:

1. **Persistent SQLite Database Record:** Run stored via `Database.save_run()` with all analytical fields populated.
2. **Standalone HTML Report:** Generated via `HTMLReportGenerator.generate_evaluation_report(run_id)`.
3. **Machine-Readable JSON Summary:** Exported metrics summary capturing all per-class dictionaries and confusion matrices.
4. **Console Log:** Full terminal execution log with run timestamps, warm-up confirmation, and repetition aggregates.

---

## 8. Protocol Governance

- **Protocol Version:** 1.0.0
- **Governing Phase:** Phase 6.1 (Protocol Design)
- **Execution Target:** Phase 6.2 (Multi-Model Benchmark Execution, upon user authorization)
