import sys
from pathlib import Path

# Add project root to sys.path FIRST to resolve imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from app.dashboard.charts import render_accuracy_trend_chart
from app.dashboard.components import (
    render_header_metrics,
    render_interactive_runner,
    render_metadata,
    render_prompt_diff_viewer,
    render_regression_status,
)
from app.dashboard.tables import (
    render_case_comparison_table,
    render_case_explorer_table,
    render_history_table,
    render_run_comparison_table,
)
from app.storage.database import Database

st.set_page_config(
    page_title="PromptOps",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 PromptOps Dashboard")

# Connect to database once
db = Database()
history = db.get_all_runs()

if not history:
    db.close()
    st.warning("No evaluation runs found.")

    # Show runner panel so a run can be triggered
    render_interactive_runner()
    st.stop()

# Sidebar
run_ids = [row[0] for row in history]
st.sidebar.header("📂 Evaluation Runs")

selected_run = st.sidebar.selectbox(
    "Choose Run",
    run_ids,
    index=0,
)
st.sidebar.markdown("---")

compare_run = st.sidebar.selectbox(
    "Compare With",
    run_ids,
    index=min(1, len(run_ids) - 1),
)

# Load selected runs details
run = db.get_run_by_id(selected_run)
comparison_run = db.get_run_by_id(compare_run)

# Latest two runs (used only for regression)
runs = db.get_last_two_runs()

# Load case results for selected runs
case_results = db.get_case_results(selected_run)
comparison_case_results = db.get_case_results(compare_run)

db.close()

if run is None:
    st.warning("No evaluation runs found.")
    st.stop()

# Render interactive runner control
render_interactive_runner()
st.write("---")

# Render summary stats and metadata
st.subheader(f"📊 Evaluation Summary (Run {selected_run})")
render_header_metrics(run)
st.write("---")
render_metadata(run)
st.write("---")

# Render evaluation history table
st.subheader("📋 Evaluation History")
history_df = render_history_table(history)
st.write("---")

# Render line chart trend
st.subheader("📈 Accuracy Trend")
render_accuracy_trend_chart(history_df)
st.write("---")

# Render comparative tables
st.subheader("📊 Run Comparison")
render_run_comparison_table(run, comparison_run, selected_run, compare_run)
st.write("---")

# Render prompt diff comparison
if comparison_run and run:
    render_prompt_diff_viewer(comparison_run[2], run[2])
    st.write("---")

# Render case-by-case comparison
st.subheader("🔍 Case-by-Case Comparison")
render_case_comparison_table(
    case_results, comparison_case_results, selected_run, compare_run
)
st.write("---")

# Render regression warnings/alerts
st.subheader("🚨 Regression Status")
render_regression_status(selected_run, run_ids, runs)
st.write("---")

# Render details on failure/success cases
st.subheader("❌ Case Explorer")
render_case_explorer_table(case_results)