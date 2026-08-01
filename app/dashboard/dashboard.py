import sys
from pathlib import Path

# 1. Add project root to sys.path FIRST to resolve the 'app' module import
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# 2. Now you can safely import from the 'app' module
import pandas as pd
import plotly.express as px
import streamlit as st

from app.reporting.regression import RegressionDetector
from app.storage.database import Database

st.set_page_config(
    page_title="PromptOps",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 PromptOps Dashboard")

# Connect to database once
db = Database()

# Get all runs for history
history = db.get_all_runs()

if not history:
    db.close()
    st.warning("No evaluation runs found.")
    st.stop()

# Sidebar
run_ids = [row[0] for row in history]

st.sidebar.header("📂 Evaluation Runs")

selected_run = st.sidebar.selectbox(
    "Choose Run",
    run_ids,
    index=0,
)

# Load selected run
run = db.get_run_by_id(selected_run)

# Latest two runs (used only for regression)
runs = db.get_last_two_runs()

# Load case results for selected run
case_results = db.get_case_results(selected_run)

db.close()

if run is None:
    st.warning("No evaluation runs found.")
    st.stop()

run_id, timestamp, prompt, model, total, passed, failed, accuracy = run

st.subheader(f"📊 Evaluation Summary (Run {run_id})")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", f"{accuracy:.2f}%")
col2.metric("Passed", passed)
col3.metric("Failed", failed)
col4.metric("Run ID", run_id)

st.write("---")

left, right = st.columns(2)

with left:
    st.markdown(f"**Prompt Version:** `{prompt}`")
    st.markdown(f"**Model:** `{model}`")

with right:
    st.markdown(f"**Timestamp:** `{timestamp}`")

st.write("---")
st.subheader("📋 Evaluation History")

history_df = pd.DataFrame(
    history,
    columns=[
        "Run ID",
        "Timestamp",
        "Prompt",
        "Model",
        "Accuracy (%)",
        "Passed",
        "Failed",
    ],
)

st.dataframe(
    history_df,
    use_container_width=True,
    hide_index=True,
)
st.write("---")
st.subheader("📈 Accuracy Trend")

chart_df = history_df.copy()

# Show oldest → newest
chart_df = chart_df.sort_values("Run ID")

fig = px.line(
    chart_df,
    x="Run ID",
    y="Accuracy (%)",
    markers=True,
    title="Accuracy Across Evaluation Runs",
)

fig.update_layout(
    xaxis_title="Run ID",
    yaxis_title="Accuracy (%)",
    yaxis_range=[0, 100],
    template="plotly_dark",
)

st.plotly_chart(
    fig,
    use_container_width=True,
)

st.write("---")
st.subheader("🚨 Regression Status")

latest_run_id = max(run_ids)

if selected_run == latest_run_id:

    if len(runs) >= 2:

        report = RegressionDetector.compare(
            previous_run=runs[1],
            current_run=runs[0],
        )

        if report["status"] == "improved":
            st.success(
                f"Accuracy Improved by {report['difference']:.2f}%"
            )

        elif report["status"] == "regression":
            st.error(
                f"Regression Detected!\n\nAccuracy dropped by {abs(report['difference']):.2f}%"
            )

        else:
            st.info("No Change in Accuracy")

    else:
        st.warning("Need at least two runs.")

else:
    st.info("Regression status is shown only for the latest evaluation run.")

st.write("---")
st.subheader("❌ Case Explorer")

if case_results:

    cases_df = pd.DataFrame(
        case_results,
        columns=[
            "Case ID",
            "Expected",
            "Predicted",
            "Passed",
            "Error"
        ]
    )

    cases_df["Status"] = cases_df["Passed"].apply(
        lambda x: "✅ PASS" if x else "❌ FAIL"
    )

    # Summary Metrics
    passed = cases_df["Passed"].sum()
    failed = len(cases_df) - passed

    c1, c2, c3 = st.columns(3)

    c1.metric("✅ Passed", passed)
    c2.metric("❌ Failed", failed)
    c3.metric("Success Rate", f"{passed/len(cases_df)*100:.0f}%")

    # Hide Error column if empty
    if cases_df["Error"].notna().any():

        display_df = cases_df[
            [
                "Case ID",
                "Expected",
                "Predicted",
                "Status",
                "Error"
            ]
        ]

    else:

        display_df = cases_df[
            [
                "Case ID",
                "Expected",
                "Predicted",
                "Status"
            ]
        ]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

else:
    st.info("No case results available.")

    