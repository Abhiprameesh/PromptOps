import sys
from pathlib import Path

# Add project root to sys.path FIRST to resolve imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from app.dashboard.charts import render_accuracy_trend_chart
from app.dashboard.components import (
    render_breakdown_section,
    render_classification_analysis,
    render_header_metrics,
    render_interactive_runner,
    render_latency_section,
    render_metadata,
    render_prompt_diff_viewer,
    render_regression_section,
)
from app.dashboard.tables import (
    render_case_comparison_table,
    render_case_explorer_table,
    render_history_table,
)
from app.reporting.regression import RegressionDetector
from app.storage.database import Database

st.set_page_config(
    page_title="PromptOps Dashboard",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 PromptOps Evaluation & Regression Platform")

# Connect to database once
db = Database()
history = db.get_all_runs()

if not history:
    db.close()
    st.warning("No evaluation runs found in the database.")
    # Show runner panel so a run can be triggered
    render_interactive_runner()
    st.stop()

# Sidebar run selection
run_ids = [row[0] for row in history]
st.sidebar.header("📂 Evaluation Runs")

selected_run = st.sidebar.selectbox(
    "Choose Run (Target)",
    run_ids,
    index=0,
)

compare_run = st.sidebar.selectbox(
    "Compare With (Baseline)",
    run_ids,
    index=min(1, len(run_ids) - 1),
)
st.sidebar.markdown("---")

# Load selected runs details (prefer extended tuples, fallback to legacy)
run = (
    db.get_extended_run_by_id(selected_run)
    if hasattr(db, "get_extended_run_by_id")
    else None
)
if run is None:
    run = db.get_run_by_id(selected_run)

comparison_run = (
    db.get_extended_run_by_id(compare_run)
    if hasattr(db, "get_extended_run_by_id")
    else None
)
if comparison_run is None:
    comparison_run = db.get_run_by_id(compare_run)

# Load case results for selected runs (prefer extended tuples)
case_results = (
    db.get_extended_case_results(selected_run)
    if hasattr(db, "get_extended_case_results")
    else None
)
if not case_results:
    case_results = db.get_case_results(selected_run)

comparison_case_results = (
    db.get_extended_case_results(compare_run)
    if hasattr(db, "get_extended_case_results")
    else None
)
if not comparison_case_results:
    comparison_case_results = db.get_case_results(compare_run)

db.close()

if run is None:
    st.warning(f"Could not load evaluation run #{selected_run}.")
    st.stop()

# Sidebar Run Info
st.sidebar.markdown(f"**Target Run:** `#{selected_run}`")
st.sidebar.markdown(f"**Baseline Run:** `#{compare_run}`")
st.sidebar.markdown(f"**Prompt Ver:** `{run[2]}`")
st.sidebar.markdown(f"**Model:** `{run[3]}`")

# 1. Interactive Runner Control
render_interactive_runner()
st.write("---")

# 2. Evaluation Overview & Metadata
st.subheader(f"📊 Evaluation Overview — Run #{selected_run}")
render_metadata(run)
st.write("")
render_header_metrics(run)
st.write("---")

# 3. Evaluation History & Trend
st.subheader("📋 Evaluation History & Accuracy Trend")
col1, col2 = st.columns([1, 1])
with col1:
    history_df = render_history_table(history)
with col2:
    render_accuracy_trend_chart(history_df)
st.write("---")

# 4. Regression Detection & Comparison Gate
st.subheader(f"🚨 Regression Analysis (Baseline #{compare_run} ➔ Target #{selected_run})")
if selected_run == compare_run:
    st.info("Baseline and target run are the same. Choose a different baseline in the sidebar to view regressions.")
else:
    try:
        report = RegressionDetector.compare(
            previous_run=comparison_run,
            current_run=run,
            previous_cases=comparison_case_results,
            current_cases=case_results,
        )
        render_regression_section(report)
    except Exception as e:
        st.error(f"Failed to perform regression comparison: {str(e)}")
st.write("---")

# 5. Classification Performance Analysis
st.subheader("🎯 Classification Performance Analysis")
cat_metrics = run[12] if len(run) > 12 and isinstance(run[12], dict) else None
pri_metrics = run[13] if len(run) > 13 and isinstance(run[13], dict) else None
render_classification_analysis(cat_metrics, pri_metrics)
st.write("---")

# 6. Difficulty & Tag Breakdown
st.subheader("🧩 Slices & Metadata Breakdowns")
diff_breakdown = run[14] if len(run) > 14 and isinstance(run[14], dict) else None
tag_breakdown = run[15] if len(run) > 15 and isinstance(run[15], dict) else None
render_breakdown_section(diff_breakdown, tag_breakdown)
st.write("---")

# 7. Latency Analysis
st.subheader("⏱️ Inference Latency Analysis")
render_latency_section(run, case_results)
st.write("---")

# 8. Prompt Diff Comparison
if comparison_run and run:
    render_prompt_diff_viewer(comparison_run[2], run[2])
    st.write("---")

# 9. Case-by-Case Comparison
st.subheader(f"🔍 Case-by-Case Comparison (Baseline #{compare_run} vs Target #{selected_run})")
render_case_comparison_table(case_results, comparison_case_results, selected_run, compare_run)
st.write("---")

# 10. Case Explorer Table
st.subheader(f"❌ Case Explorer (Run #{selected_run})")
render_case_explorer_table(case_results)
