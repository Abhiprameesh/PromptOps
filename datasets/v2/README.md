# PromptOps Golden Dataset V2: Specification & Documentation

## 1. Dataset Purpose

The **PromptOps Golden Dataset V2** (`datasets/v2/github_issues.json`) is an expanded, production-representative evaluation dataset designed to rigorously assess LLM performance on GitHub issue triage and classification. 

While the initial V1 dataset (`GH001`–`GH005`) served as an essential smoke-test and unit-validation suite, its small sample size (5 cases) and partial category coverage (only 5 of 8 supported categories) prevented statistically reliable model-to-model comparisons. Dataset V2 expands the golden suite to **50 meticulously curated test cases**, establishing a balanced, multi-category, multi-priority benchmark suitable for reproducible evaluations across local and hosted LLMs.

---

## 2. Dataset Version & Metadata

- **Dataset File:** `datasets/v2/github_issues.json`
- **Version Identifier:** `v2`
- **Creation Date:** `2026-11-23`
- **Total Test Cases:** 50 (`GH001` through `GH050`)
- **Schema Compliance:** Validated against `app.evaluation.schema.GoldenDataset` (Pydantic V2)
- **Taxonomy Source:** Aligned with `prompt/v1.yaml` and `app.core.schemas.IssueClassification`

---

## 3. Dataset Distributions

### A. Category Distribution (8 Categories)

Every supported category in the PromptOps taxonomy is represented with balanced sample counts:

| Category | Count | Percentage | Primary Focus Area |
| :--- | :---: | :---: | :--- |
| `backend` | 7 | 14.0% | Server logic, API crashes, serialization, webhooks, worker queues |
| `bug` | 7 | 14.0% | Functional errors, runtime exceptions, metric bugs, race conditions |
| `authentication` | 6 | 12.0% | OAuth SSO, JWT expiration, session invalidation, MFA, cookies |
| `database` | 6 | 12.0% | Migrations, connection pools, deadlocks, composite indexes, cascades |
| `documentation` | 6 | 12.0% | Outdated guides, broken links, quickstarts, Kubernetes env vars, Swagger |
| `feature-request`| 6 | 12.0% | Export capabilities, Slack alerts, model prompt configs, search filters |
| `performance` | 6 | 12.0% | Slow dashboards, N+1 queries, memory growth, regex backtracking, caching |
| `ui` | 6 | 12.0% | Safari menu hover, button states, modal viewport traps, contrast, pagination |
| **Total** | **50** | **100.0%** | |

### B. Priority Distribution (3 Levels)

The dataset avoids the heavy high-priority skew of V1 by including balanced realistic priorities:

| Priority | Count | Percentage | Operational Definition |
| :--- | :---: | :---: | :--- |
| `high` | 18 | 36.0% | Outages, data loss, security vulnerabilities, fatal exceptions, blockers |
| `medium` | 20 | 40.0% | Functional defects, significant UX flaws, important enhancements, missing features |
| `low` | 12 | 24.0% | Cosmetic flaws, typos, minor documentation fixes, non-urgent feature requests |
| **Total** | **50** | **100.0%** | |

### C. Difficulty Distribution (3 Tiers)

Difficulty is assigned based on classification ambiguity and contextual nuance, rather than text length:

| Difficulty | Count | Percentage | Description |
| :--- | :---: | :---: | :--- |
| `easy` | 21 | 42.0% | Direct, unambiguous descriptions with explicit keywords and clear root causes |
| `medium` | 20 | 40.0% | Realistic multi-sentence bug reports requiring technical context and domain deduction |
| `hard` | 9 | 18.0% | Overlapping symptoms (e.g. database deadlocks causing 500 API errors) |
| **Total** | **50** | **100.0%** | |

### D. Tag Distribution (Top Tags)

The dataset features 118 unique descriptive tags. Core domain tags appear with high frequency:
- `authentication`: 7 cases
- `backend`: 7 cases
- `ui`: 7 cases
- `bug`: 6 cases
- `database`: 6 cases
- `documentation`: 6 cases
- `performance`: 6 cases
- `api`: 5 cases
- `feature-request`: 5 cases
- `dashboard`: 3 cases
- `frontend`: 3 cases
- `cli`, `css`, `error-handling`, `export`, `locking`, `memory`, `optimization`, `schema`, `session`, `sql`, `sqlite`: 2 cases each
- 96 specialized domain tags (e.g., `oauth`, `alembic`, `jwt`, `celery`, `deadlock`, `regex`, `modal`): 1 case each

---

## 4. Labeling Methodology

1. **Manual Ground-Truth Assignment:**
   No synthetic or automated LLM labeling was used to generate ground-truth labels. All 50 test cases were authored and manually curated by domain experts to ensure defensibility.
2. **Strict Taxonomy Adherence:**
   Labels strictly utilize the allowed literal values in `IssueClassification`:
   - Categories: `authentication`, `ui`, `backend`, `database`, `performance`, `documentation`, `bug`, `feature-request`.
   - Priorities: `low`, `medium`, `high`.
3. **One Single Canonical Ground Truth:**
   Even when an issue exhibits multi-layered symptoms, exactly one canonical category and priority is established based on root-cause attribution.
4. **Defensible Documentation in Notes:**
   Every single case includes an explanatory `notes` field in its metadata articulating the rationale behind the chosen classification.

---

## 5. Handling of Ambiguous & Overlapping Cases

Software engineering issues often exhibit symptoms in one layer while having their root cause in another. V2 intentionally includes realistic overlapping cases, resolving them through documented engineering rules:

| Case ID | Symptom / Description | Competing Labels | Assigned Category | Resolution Rationale |
| :--- | :--- | :--- | :--- | :--- |
| `GH010` | Stripe webhook returns 401 Unauthorized | `backend` vs `authentication` | `authentication` | Root cause is auth middleware incorrectly intercepting public webhook routes. |
| `GH014` | Modal dialog extends beyond viewport | `ui` vs `bug` | `ui` | Frontend CSS viewport constraint failure. |
| `GH017` | Pagination overflows on 10,000+ items | `performance` vs `ui` | `ui` | Rendering and layout overflow, not backend query delay. |
| `GH020` | Concurrent transactions trigger deadlocks | `database` vs `backend` | `database` | PostgreSQL transaction lock ordering and concurrency failure. |
| `GH029` | Payment gateway TLS handshake timeout | `backend` vs `integration` | `backend` | Missing retry and timeout handling in backend HTTP client. |
| `GH032` | Regex backtracking freezes event loop | `bug` vs `performance` | `performance` | CPU exhaustion caused by polynomial regular expression backtracking. |
| `GH038` | Swagger schema lists outdated params | `backend` vs `documentation` | `documentation` | Swagger UI is an API reference document; the endpoint behavior itself is correct. |
| `GH042` | Custom system prompts per model family | `feature-request` vs `prompt` | `feature-request` | Enhancement request to expand prompt configuration schema. |
| `GH049` | CLI exits with code 0 on missing DB | `bug` vs `cli` | `bug` | Logic defect causing silent failure in CI gating script. |
| `GH050` | SQLite database locked under async load | `database` vs `bug` | `bug` | Defective connection concurrency handling in application runner code. |

---

## 6. Relationship Between V1 and V2

- **V1 Dataset (`datasets/v1/github_issues.json`):**
  - Remained 100% untouched and preserved.
  - Exactly 5 test cases (`GH001`–`GH005`).
  - Represents historical evaluations (runs 1–5).
- **V2 Dataset (`datasets/v2/github_issues.json`):**
  - Directly incorporates `GH001`–`GH005` as cases 1 through 5 with identical titles, bodies, expected classifications, and metadata.
  - Extends the dataset with 45 new cases (`GH006` through `GH050`).
  - Guarantees backward compatibility: any evaluation run targeting V2 can trace performance on the original V1 subset while evaluating against the expanded 50-case benchmark.

---

## 7. Known Dataset Limitations

1. **Synthetic Golden Dataset:**
   While descriptions replicate real-world GitHub issues, they are curated examples rather than scraped telemetry from production repositories.
2. **Deterministic Single-Label Assumption:**
   In real-world triage, an issue may legitimately be labeled with multiple tags (e.g., both `backend` and `database`). The benchmark currently requires single-category classification to compute precision, recall, and confusion matrices.
3. **English Language Only:**
   All issue descriptions and expected summaries are written in English. Multilingual issue triage is not covered in V2.
4. **Summary Evaluation Strictness:**
   V2 evaluates summary validity based on presence and non-empty content. Semantic summary distance (e.g. ROUGE, BERTScore) is not currently part of the deterministic pass condition.

---

## 8. Dataset Quality Summary

- **Total Cases:** 50
- **Missing / Null Required Fields:** 0
- **Schema Violations:** 0
- **Duplicate Case IDs:** 0
- **Duplicate Issue Descriptions:** 0
- **Category Coverage:** 8 / 8 categories represented (min 6 cases/category)
- **Priority Coverage:** 3 / 3 priorities represented (min 12 cases/priority)
- **Difficulty Coverage:** 3 / 3 difficulty tiers represented (min 9 cases/tier)
- **Validation Status:** Automated test suite `tests/test_dataset_v2.py` passing 100% (9/9 tests).
