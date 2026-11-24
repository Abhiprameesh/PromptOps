# PromptOps Benchmark V2 Comparative Analysis Specification

## 1. Executive Summary

This directory contains the official comparative evaluation artifacts for the **PromptOps Phase 6.3 Multi-Model Benchmark**.
The evaluation was conducted on the **Expanded Golden Dataset V2** (50 test cases, 3 repetitions per model, 450 total measured inferences) comparing:
- **Gemma 3 4B** (`gemma3:4b`, Runs #10, #11, #12)
- **Mistral 7B** (`mistral:latest`, Runs #13, #14, #15)
- **Llama 3 8B** (`llama3:latest`, Runs #16, #17, #18)

---

## 2. Directory Artifacts

| Filename | Type | Purpose |
| :--- | :---: | :--- |
| `comparison_summary.json` | JSON | Complete machine-readable metrics, rankings, and deployment recommendations. |
| `model_comparison.csv` | CSV | High-level model comparison table across all classification and latency metrics. |
| `per_class_comparison.csv` | CSV | Granular precision, recall, and F1 scores for each category and priority class. |
| `latency_comparison.csv` | CSV | Per-repetition and aggregate latency statistics (mean, median, P95, min, max, std dev). |
| `repetition_consistency.json` | JSON | Consistency analysis across repetitions 1, 2, and 3. |
| `reliability_comparison.json` | JSON | Reliability, parsing, and JSON validity metrics (100% success rate across 450 inferences). |
| `confusion_matrices.json` | JSON | Full $8 \times 9$ category and $3 \times 4$ priority confusion matrices and top confused pairs. |
| `comparative_report.html` | HTML | Standalone, responsive HTML visualization with embedded CSS and SVG charts. |

---

## 3. High-Level Findings & Rankings

1. **Overall Composite Accuracy:**
   - **Gemma 3 4B:** 32.00% (Rank #1)
   - **Llama 3 8B:** 26.00% (Rank #2)
   - **Mistral 7B:** 24.00% (Rank #3)
2. **Category Classification Accuracy & Macro F1:**
   - **Mistral 7B:** 70.00% (Macro F1: 0.6894, Rank #1)
   - **Llama 3 8B:** 62.00% (Macro F1: 0.5970, Rank #2)
   - **Gemma 3 4B:** 48.00% (Macro F1: 0.4357, Rank #3)
3. **Priority Classification Accuracy:**
   - **Gemma 3 4B:** 52.00% (Rank #1)
   - **Llama 3 8B:** 44.00% (Rank #2)
   - **Mistral 7B:** 38.00% (Rank #3)

---

## 4. Evidence-Based Deployment Recommendations

- **Autonomous End-to-End Triage:** Deploy **Gemma 3 4B**. Achieves the highest composite success rate (32.00%) and lowest mean latency.
- **Queue/Department Routing:** Deploy **Mistral 7B**. Achieves the highest category classification accuracy (70.00%) with superior F1 in technical backend, database, and UI tickets.
