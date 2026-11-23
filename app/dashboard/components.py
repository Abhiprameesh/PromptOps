import asyncio
import difflib
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from app.core.config import load_prompt_config
from app.dashboard.charts import (
    render_breakdown_bar_chart,
    render_confusion_matrix_chart,
    render_latency_chart,
)
from app.dashboard.tables import (
    prepare_breakdown_tables,
    prepare_confusion_matrix_df,
    render_classification_table,
)
from app.evaluation.loader import load_dataset
from app.evaluation.runner import EvaluationRunner
from app.reporting.html_report import HTMLReportGenerator
from app.reporting.regression import RegressionDetector, RegressionThresholds
from app.storage.database import Database


def prepare_overview_metrics(run: Any) -> Dict[str, Any]:
    """
    Extracts and formats all V1 and V2 summary metrics from a run tuple.
    Handles 8-element (legacy V1) and 16-element (V2) tuples safely.
    Missing metrics are formatted as 'N/A' (never 0%).
    """
    if run is None:
        return {}

    run_id = run[0]
    timestamp = run[1]
    prompt = run[2]
    model = run[3]
    total_cases = run[4]
    passed = run[5]
    failed = run[6]
    accuracy = run[7]

    is_v2 = len(run) >= 16

    cat_acc = run[8] if is_v2 and run[8] is not None else None
    pri_acc = run[9] if is_v2 and run[9] is not None else None
    comp_acc = run[10] if is_v2 and run[10] is not None else None
    avg_lat = run[11] if is_v2 and run[11] is not None else None

    # Classification metrics
    cat_metrics = run[12] if is_v2 and isinstance(run[12], dict) else {}
    pri_metrics = run[13] if is_v2 and isinstance(run[13], dict) else {}

    cat_macro_f1 = (
        cat_metrics.get("macro_avg", {}).get("f1")
        if isinstance(cat_metrics, dict)
        else None
    )
    cat_weighted_f1 = (
        cat_metrics.get("weighted_avg", {}).get("f1")
        if isinstance(cat_metrics, dict)
        else None
    )
    pri_macro_f1 = (
        pri_metrics.get("macro_avg", {}).get("f1")
        if isinstance(pri_metrics, dict)
        else None
    )

    return {
        "run_id": run_id,
        "timestamp": timestamp,
        "prompt": prompt,
        "model": model,
        "total_cases": total_cases,
        "passed": passed,
        "failed": failed,
        "raw_accuracy": accuracy,
        "accuracy": f"{accuracy:.2f}%" if accuracy is not None else "N/A",
        "category_accuracy": f"{cat_acc:.2f}%" if cat_acc is not None else "N/A",
        "priority_accuracy": f"{pri_acc:.2f}%" if pri_acc is not None else "N/A",
        "composite_accuracy": f"{comp_acc:.2f}%" if comp_acc is not None else "N/A",
        "category_macro_f1": f"{cat_macro_f1:.4f}" if cat_macro_f1 is not None else "N/A",
        "category_weighted_f1": f"{cat_weighted_f1:.4f}" if cat_weighted_f1 is not None else "N/A",
        "priority_macro_f1": f"{pri_macro_f1:.4f}" if pri_macro_f1 is not None else "N/A",
        "avg_latency_ms": f"{avg_lat:.2f} ms" if avg_lat is not None else "N/A",
        "raw_latency": avg_lat,
    }


def render_header_metrics(run: Any):
    """
    Renders comprehensive V1 and V2 summary metric cards for the selected run.
    """
    metrics = prepare_overview_metrics(run)
    if not metrics:
        st.warning("No metrics available for this run.")
        return

    # Row 1: Accuracies
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall Accuracy", metrics["accuracy"])
    c2.metric("Category Accuracy", metrics["category_accuracy"])
    c3.metric("Priority Accuracy", metrics["priority_accuracy"])
    c4.metric("Composite Accuracy", metrics["composite_accuracy"])

    # Row 2: F1 & Latency
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Category Macro F1", metrics["category_macro_f1"])
    c6.metric("Category Weighted F1", metrics["category_weighted_f1"])
    c7.metric("Priority Macro F1", metrics["priority_macro_f1"])
    c8.metric("Avg Latency", metrics["avg_latency_ms"])

    # Row 3: Execution Counts
    c9, c10, c11, c12 = st.columns(4)
    c9.metric("Total Cases", metrics["total_cases"])
    c10.metric("Passed Cases", metrics["passed"])
    c11.metric("Failed Cases", metrics["failed"])
    c12.metric("Run ID", f"#{metrics['run_id']}")


def render_metadata(run: Any):
    """
    Renders the prompt, model, and timestamp details of the run.
    """
    metrics = prepare_overview_metrics(run)
    if not metrics:
        return

    left, right = st.columns(2)
    with left:
        st.markdown(f"**Prompt Version:** `{metrics['prompt']}`")
        st.markdown(f"**Model:** `{metrics['model']}`")
    with right:
        st.markdown(f"**Timestamp:** `{metrics['timestamp']}`")
        st.markdown(f"**Total Cases Evaluated:** `{metrics['total_cases']}`")


def render_classification_analysis(
    category_metrics: Optional[Dict[str, Any]],
    priority_metrics: Optional[Dict[str, Any]],
):
    """
    Renders separate category and priority classification analysis sections
    including per-class precision, recall, F1, support, and confusion matrices.
    """
    cat_tab, pri_tab = st.tabs(["🏷️ Category Classification", "⚡ Priority Classification"])

    with cat_tab:
        if category_metrics and isinstance(category_metrics, dict) and category_metrics.get("per_class"):
            # Metric cards
            macro_f1 = category_metrics.get("macro_avg", {}).get("f1")
            weighted_f1 = category_metrics.get("weighted_avg", {}).get("f1")
            cat_acc = category_metrics.get("accuracy")

            m1, m2, m3 = st.columns(3)
            m1.metric("Category Accuracy", f"{cat_acc:.2f}%" if cat_acc is not None else "N/A")
            m2.metric("Macro F1", f"{macro_f1:.4f}" if macro_f1 is not None else "N/A")
            m3.metric("Weighted F1", f"{weighted_f1:.4f}" if weighted_f1 is not None else "N/A")

            st.write("##### Per-Class Performance")
            render_classification_table(category_metrics)

            st.write("##### Category Confusion Matrix")
            cm_df = prepare_confusion_matrix_df(category_metrics)
            if cm_df is not None and not cm_df.empty:
                render_confusion_matrix_chart(cm_df, title="Category Confusion Matrix (Rows: True, Cols: Predicted)")
            else:
                st.info("No confusion matrix available.")
        else:
            st.info("Category classification metrics are unavailable for this run (legacy V1 run).")

    with pri_tab:
        if priority_metrics and isinstance(priority_metrics, dict) and priority_metrics.get("per_class"):
            # Metric cards
            macro_f1 = priority_metrics.get("macro_avg", {}).get("f1")
            weighted_f1 = priority_metrics.get("weighted_avg", {}).get("f1")
            pri_acc = priority_metrics.get("accuracy")

            m1, m2, m3 = st.columns(3)
            m1.metric("Priority Accuracy", f"{pri_acc:.2f}%" if pri_acc is not None else "N/A")
            m2.metric("Macro F1", f"{macro_f1:.4f}" if macro_f1 is not None else "N/A")
            m3.metric("Weighted F1", f"{weighted_f1:.4f}" if weighted_f1 is not None else "N/A")

            st.write("##### Per-Class Performance")
            render_classification_table(priority_metrics)

            st.write("##### Priority Confusion Matrix")
            cm_df = prepare_confusion_matrix_df(priority_metrics)
            if cm_df is not None and not cm_df.empty:
                render_confusion_matrix_chart(cm_df, title="Priority Confusion Matrix (Rows: True, Cols: Predicted)")
            else:
                st.info("No confusion matrix available.")
        else:
            st.info("Priority classification metrics are unavailable for this run (legacy V1 run).")


def prepare_case_churn_summary(case_analysis: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Summarizes case-level churn breakdown into counts and lists.
    """
    if not case_analysis or not isinstance(case_analysis, dict):
        return {
            "newly_failed": [],
            "recovered": [],
            "unchanged_passing": [],
            "unchanged_failing": [],
            "added_cases": [],
            "removed_cases": [],
            "missing_pass_cases": [],
            "duplicate_cases": [],
        }

    return {
        "newly_failed": list(case_analysis.get("newly_failed", [])),
        "recovered": list(case_analysis.get("recovered", [])),
        "unchanged_passing": list(case_analysis.get("unchanged_passing", [])),
        "unchanged_failing": list(case_analysis.get("unchanged_failing", [])),
        "added_cases": list(case_analysis.get("added_cases", [])),
        "removed_cases": list(case_analysis.get("removed_cases", [])),
        "missing_pass_cases": list(case_analysis.get("missing_pass_cases", [])),
        "duplicate_cases": list(case_analysis.get("duplicate_cases", [])),
    }


def render_case_churn_view(case_analysis: Optional[Dict[str, Any]]):
    """
    Renders case-level churn counts and expandable details for all categories.
    """
    churn = prepare_case_churn_summary(case_analysis)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔴 Newly Failed", len(churn["newly_failed"]))
    c2.metric("🟢 Recovered", len(churn["recovered"]))
    c3.metric("⚪ Unchanged Passing", len(churn["unchanged_passing"]))
    c4.metric("⚪ Unchanged Failing", len(churn["unchanged_failing"]))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("➕ Added Cases", len(churn["added_cases"]))
    c6.metric("➖ Removed Cases", len(churn["removed_cases"]))
    c7.metric("⚠️ Missing Status", len(churn["missing_pass_cases"]))
    c8.metric("⚠️ Duplicates", len(churn["duplicate_cases"]))

    # Detail Expanders
    if churn["newly_failed"]:
        with st.expander(f"🔴 Newly Failed Cases ({len(churn['newly_failed'])})", expanded=True):
            st.write(", ".join(f"`{c}`" for c in churn["newly_failed"]))

    if churn["recovered"]:
        with st.expander(f"🟢 Recovered Cases ({len(churn['recovered'])})", expanded=False):
            st.write(", ".join(f"`{c}`" for c in churn["recovered"]))

    if churn["added_cases"]:
        with st.expander(f"➕ Newly Added Cases ({len(churn['added_cases'])})", expanded=False):
            st.write(", ".join(f"`{c}`" for c in churn["added_cases"]))

    if churn["removed_cases"]:
        with st.expander(f"➖ Removed Cases ({len(churn['removed_cases'])})", expanded=False):
            st.write(", ".join(f"`{c}`" for c in churn["removed_cases"]))

    if churn["missing_pass_cases"]:
        with st.expander(f"⚠️ Cases with Missing Status ({len(churn['missing_pass_cases'])})", expanded=False):
            st.write(", ".join(f"`{c}`" for c in churn["missing_pass_cases"]))

    if churn["duplicate_cases"]:
        with st.expander(f"⚠️ Detected Duplicate Cases ({len(churn['duplicate_cases'])})", expanded=False):
            st.write(", ".join(f"`{c}`" for c in churn["duplicate_cases"]))


def render_regression_section(report: Dict[str, Any]):
    """
    Renders comprehensive regression analysis comparing baseline and current run.
    """
    if not report:
        st.info("No regression comparison report available.")
        return

    severity = report.get("severity", "none")
    status = report.get("status", "unchanged")
    action = report.get("recommended_action", "")

    # Status Banner
    if severity == "critical":
        st.error(f"### 🔴 CRITICAL REGRESSION DETECTED\n**Action:** {action}")
    elif severity == "warning":
        st.warning(f"### 🟡 WARNING: PERFORMANCE DEGRADATION\n**Action:** {action}")
    elif status == "improved":
        st.success(f"### 🟢 PERFORMANCE IMPROVED\n**Action:** {action}")
    else:
        st.info(f"### ⚪ NO REGRESSION DETECTED\n**Action:** {action}")

    # Metrics comparison table
    st.write("#### 📊 Key Metrics Comparison")
    metrics_comp = report.get("metrics_comparison", {})
    if metrics_comp:
        rows = []
        for name, data in metrics_comp.items():
            prev_val = data.get("previous")
            curr_val = data.get("current")
            delta = data.get("delta")
            m_status = data.get("status", "unavailable")

            is_lat = "latency" in name
            fmt = "{:.2f} ms" if is_lat else "{:.2f}%" if "accuracy" in name else "{:.4f}"

            prev_str = fmt.format(prev_val) if prev_val is not None else "N/A"
            curr_str = fmt.format(curr_val) if curr_val is not None else "N/A"

            if delta is not None:
                sign = "+" if delta > 0 else ""
                delta_str = f"{sign}{fmt.format(delta)}"
            else:
                delta_str = "N/A"

            status_icon = (
                "🔴 Critical"
                if m_status == "critical"
                else "🟡 Warning"
                if m_status == "warning"
                else "🟢 Improved"
                if m_status == "improved"
                else "⚪ Unchanged"
                if m_status == "unchanged"
                else "⚪ N/A"
            )

            rows.append(
                {
                    "Metric": name,
                    f"Baseline (Run #{report.get('previous_run')})": prev_str,
                    f"Current (Run #{report.get('current_run')})": curr_str,
                    "Delta": delta_str,
                    "Status": status_icon,
                }
            )

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Degraded / Improved Callouts
    degraded = report.get("degraded_metrics", [])
    improved = report.get("improved_metrics", [])

    if degraded or improved:
        col_d, col_i = st.columns(2)
        with col_d:
            if degraded:
                st.write("##### ⚠️ Degraded Metrics & Classes")
                for d in degraded:
                    m = d.get("metric")
                    sev = d.get("severity", "warning").upper()
                    delta = d.get("delta")
                    st.markdown(f"- **{m}**: `{delta}` ({sev})")
            else:
                st.write("##### ⚠️ Degraded Metrics")
                st.write("No metrics degraded.")

        with col_i:
            if improved:
                st.write("##### 🟢 Improved Metrics & Classes")
                for imp in improved:
                    m = imp.get("metric")
                    delta = imp.get("delta")
                    st.markdown(f"- **{m}**: `+{delta}`")
            else:
                st.write("##### 🟢 Improved Metrics")
                st.write("No metrics improved.")

    # Case Churn View
    st.write("#### 🔍 Case-Level Churn Analysis")
    render_case_churn_view(report.get("case_analysis"))


def render_breakdown_section(
    difficulty_breakdown: Optional[Dict[str, Any]],
    tag_breakdown: Optional[Dict[str, Any]],
):
    """
    Renders slice breakdowns by difficulty level and tags.
    """
    diff_df, tag_df = prepare_breakdown_tables(difficulty_breakdown, tag_breakdown)

    col1, col2 = st.columns(2)
    with col1:
        st.write("##### 🧩 Difficulty Breakdown")
        if diff_df is not None and not diff_df.empty:
            st.dataframe(diff_df, use_container_width=True, hide_index=True)
            render_breakdown_bar_chart(diff_df, "Difficulty Level", "Accuracy by Difficulty")
        else:
            st.info("No difficulty breakdown available for this run (legacy V1 run or not specified).")

    with col2:
        st.write("##### 🏷️ Tag Breakdown")
        if tag_df is not None and not tag_df.empty:
            st.dataframe(tag_df, use_container_width=True, hide_index=True)
            render_breakdown_bar_chart(tag_df, "Tag", "Accuracy by Tag")
        else:
            st.info("No tag breakdown available for this run (legacy V1 run or not specified).")


def render_latency_section(run: Any, case_results: List[Any]):
    """
    Renders average and per-case inference latency analysis.
    """
    metrics = prepare_overview_metrics(run)
    raw_lat = metrics.get("raw_latency")

    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric("Avg Latency", metrics.get("avg_latency_ms", "N/A"))
        if raw_lat is not None:
            st.caption("Measured via high-resolution timer (`time.perf_counter`) around LLM inference.")

    with col2:
        render_latency_chart(case_results, avg_latency=raw_lat)


def render_regression_status(selected_run: int, run_ids: List[int], runs: List[Any]):
    """
    Backward-compatible regression status checker for the latest run.
    """
    latest_run_id = max(run_ids)

    if selected_run == latest_run_id:
        if len(runs) >= 2:
            report = RegressionDetector.compare(
                previous_run=runs[1],
                current_run=runs[0],
            )
            render_regression_section(report)
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
        st.info(f"Both runs used prompt version `{prompt_version_1}`. No diff to show.")
        return

    path1 = Path("prompt") / f"{prompt_version_1}.yaml"
    path2 = Path("prompt") / f"{prompt_version_2}.yaml"

    if not path1.exists() or not path2.exists():
        st.warning(
            f"Could not locate prompt files for comparison (`{prompt_version_1}.yaml` or `{prompt_version_2}.yaml`)."
        )
        return

    try:
        with open(path1, "r", encoding="utf-8") as f1, open(path2, "r", encoding="utf-8") as f2:
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
    dataset_files = [str(d.relative_to(dataset_dir)) for d in dataset_dir.glob("**/*.json")]

    if not prompt_files or not dataset_files:
        st.warning("Ensure at least one prompt (YAML) and one dataset (JSON) exist.")
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
            status_text.text(f"Running evaluation on {len(dataset.cases)} cases in parallel...")

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
