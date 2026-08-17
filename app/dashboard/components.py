import asyncio
import difflib
import os
from pathlib import Path
from typing import Any, List

import streamlit as st

from app.core.config import load_prompt_config
from app.evaluation.loader import load_dataset
from app.evaluation.runner import EvaluationRunner
from app.reporting.html_report import HTMLReportGenerator
from app.reporting.regression import RegressionDetector
from app.storage.database import Database


def render_header_metrics(run: Any):
    """
    Renders summary metric cards for the selected run.
    """
    run_id, _, _, _, _, passed, failed, accuracy = run

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy:.2f}%")
    col2.metric("Passed", passed)
    col3.metric("Failed", failed)
    col4.metric("Run ID", run_id)


def render_metadata(run: Any):
    """
    Renders the prompt, model, and timestamp details of the run.
    """
    _, timestamp, prompt, model, _, _, _, _ = run

    left, right = st.columns(2)
    with left:
        st.markdown(f"**Prompt Version:** `{prompt}`")
        st.markdown(f"**Model:** `{model}`")
    with right:
        st.markdown(f"**Timestamp:** `{timestamp}`")


def render_regression_status(selected_run: int, run_ids: List[int], runs: List[Any]):
    """
    Checks and displays warning/success banners for model regressions.
    """
    latest_run_id = max(run_ids)

    if selected_run == latest_run_id:
        if len(runs) >= 2:
            report = RegressionDetector.compare(
                previous_run=runs[1],
                current_run=runs[0],
            )

            if report["status"] == "improved":
                st.success(f"Accuracy Improved by {report['difference']:.2f}%")
            elif report["status"] == "regression":
                st.error(
                    f"Regression Detected!\n\nAccuracy dropped by {abs(report['difference']):.2f}%"
                )
            else:
                st.info("No Change in Accuracy")
        else:
            st.warning("Need at least two runs to detect regression.")
    else:
        st.info("Regression status is shown only for the latest evaluation run.")


def render_prompt_diff_viewer(prompt_version_1: str, prompt_version_2: str):
    """
    Loads prompt configs from disk and displays a code diff comparing them.
    """
    st.subheader("📝 Prompt Diff Viewer")
    if prompt_version_1 == prompt_version_2:
        st.info(
            f"Both runs used prompt version `{prompt_version_1}`. No diff to show."
        )
        return

    path1 = Path("prompt") / f"{prompt_version_1}.yaml"
    path2 = Path("prompt") / f"{prompt_version_2}.yaml"

    if not path1.exists() or not path2.exists():
        st.warning(
            f"Could not locate prompt files for comparison (`{prompt_version_1}.yaml` or `{prompt_version_2}.yaml`)."
        )
        return

    try:
        with open(path1, "r", encoding="utf-8") as f1, open(
            path2, "r", encoding="utf-8"
        ) as f2:
            lines1 = f1.readlines()
            lines2 = f2.readlines()

        diff = list(
            difflib.unified_diff(
                lines1,
                lines2,
                fromfile=f"{prompt_version_1}.yaml",
                tofile=f"{prompt_version_2}.yaml",
            )
        )

        if not diff:
            st.info("The prompt contents are identical.")
        else:
            diff_text = "".join(diff)
            st.code(diff_text, language="diff")
    except Exception as e:
        st.error(f"Failed to read prompt files for diff: {str(e)}")


def render_interactive_runner():
    """
    Renders the evaluation run trigger panel inside Streamlit.
    """
    st.subheader("⚡ Run New Evaluation")

    prompt_dir = Path("prompt")
    prompt_files = [p.name for p in prompt_dir.glob("*.yaml")]

    dataset_dir = Path("datasets")
    dataset_files = [
        str(d.relative_to(dataset_dir))
        for d in dataset_dir.glob("**/*.json")
    ]

    if not prompt_files or not dataset_files:
        st.warning(
            "Ensure at least one prompt (YAML) and one dataset (JSON) exist."
        )
        return

    col1, col2 = st.columns(2)
    with col1:
        selected_prompt = st.selectbox("Select Prompt Configuration", prompt_files)
    with col2:
        selected_dataset = st.selectbox("Select Golden Dataset", dataset_files)

    if st.button("🚀 Start Evaluation Run"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.text("Loading configurations...")
            progress_bar.progress(20)

            config_path = prompt_dir / selected_prompt
            dataset_path = dataset_dir / selected_dataset

            config = load_prompt_config(str(config_path))
            dataset = load_dataset(str(dataset_path))

            progress_bar.progress(40)
            status_text.text(
                f"Running evaluation on {len(dataset.cases)} cases in parallel..."
            )

            runner = EvaluationRunner()

            # Safeguard running asyncio inside streamlit thread
            try:
                result = asyncio.run(runner.evaluate(dataset, config))
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(runner.evaluate(dataset, config))
                loop.close()

            progress_bar.progress(70)
            status_text.text("Saving results to database...")

            db = Database()
            run_id = db.save_run(
                result=result,
                prompt_version=config.version,
                model=config.model,
            )

            status_text.text("Generating HTML report...")
            progress_bar.progress(90)

            report_path = HTMLReportGenerator.generate(
                run_id=run_id,
                result=result,
                prompt_version=config.version,
                model=config.model,
            )

            db.close()

            progress_bar.progress(100)
            status_text.empty()
            st.success(
                f"Run {run_id} completed! Accuracy: {result.accuracy:.2f}%. Report generated at `{report_path}`."
            )

            # Rerun the Streamlit page to show the new run
            st.rerun()

        except Exception as e:
            status_text.empty()
            progress_bar.empty()
            st.error(f"Failed to run evaluation: {str(e)}")
