# PromptOps 🤖

An automated LLM Prompt Evaluation & Regression Detection platform. Define prompts as code, evaluate LLM outputs against a golden dataset locally with Ollama, compare prompt versions, and track performance changes using a clean Streamlit dashboard.

---

## Features
* **YAML-Based Prompts**: Declare prompt text, system instructions, and generation parameters in version-controlled config files (e.g., `prompt/v1.yaml`).
* **Parallel Local Inference**: Evaluates multiple test cases concurrently using an asynchronous Ollama client.
* **Regression Detection**: Keeps evaluation histories in a local SQLite database and monitors if prompt revisions lead to accuracy degradation.
* **Streamlit Dashboard**: A modular dashboard for visualizing runs history, comparing runs side-by-side, inspecting test case failures, and viewing prompt diffs.
* **Interactive Evaluation**: Select prompts and golden datasets directly from the Streamlit UI to trigger new evaluations.
* **FastAPI Server**: Serves a `/infer` endpoint to deploy your triaged prompts for production usage.

---

## Project Structure
```text
PromptOps/
├── app/
│   ├── api/             # FastAPI endpoint routing
│   ├── core/            # Configuration schemas & prompt loaders
│   ├── dashboard/       # Streamlit dashboard modules (charts, tables, components)
│   ├── evaluation/      # Dataset loading and parallel runner logic
│   ├── llm/             # Ollama client connection & output parsing
│   ├── reporting/       # Regression detection & HTML report generation
│   ├── services/        # Inference orchestrators
│   └── storage/         # SQLite database connector
├── datasets/            # Versioned golden datasets
├── prompt/              # YAML prompt configurations
├── scripts/             # Automation testing & debugging scripts
└── tests/               # Unit and integration test suites
```

---

## Prerequisites
1. **Python**: Python 3.11+ is required.
2. **Ollama**: Download and install [Ollama](https://ollama.com) locally.
3. **Download Model**: Pull the default LLM used by PromptOps:
   ```bash
   ollama pull gemma3:4b
   ```

---

## Installation & Setup

1. **Activate the Virtual Environment** (Windows):
   ```powershell
   .venv\Scripts\activate
   ```
   *(If you are setting it up fresh, run `python -m venv .venv` and activate it).*

2. **Install Dependencies**:
   ```bash
   .venv\Scripts\python -m pip install -r requirements.txt
   ```

---

## How to Run

### 1. Run Evaluations via Terminal
To execute a prompt evaluation run against the golden dataset and save results to the database:
```bash
.venv\Scripts\python -m scripts.test_runner
```
This saves the run data, runs a regression check, and writes a detailed HTML report file under `reports/`.

### 2. Launch the Streamlit Dashboard
To open the interactive runs visualizer, compare prompts, view diffs, and trigger evaluations:
```bash
.venv\Scripts\streamlit run app/dashboard/dashboard.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 3. Start the FastAPI Server
To run the server hosting the issue triage API endpoint:
```bash
.venv\Scripts\uvicorn main:app --reload
```
You can send test issues via POST request to `http://127.0.0.1:8000/infer/` with JSON:
```json
{
  "title": "Cannot login",
  "body": "User receives a 500 error after clicking submit button."
}
```

### 4. Run Unit Tests
To run the test suite verifying configuration loading, parser behavior, and regression logic:
```bash
.venv\Scripts\python -m unittest discover -s tests
```