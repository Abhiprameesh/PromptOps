import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import pandas as pd
from app.reporting.regression import RegressionDetector
# Add project root to sys.path to resolve the 'app' module import
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
from app.storage.database import Database

st.set_page_config(
    page_title="PromptOps",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 PromptOps Dashboard")

# Connect to database once
db = Database()

latest = db.get_latest_run()
history = db.get_all_runs()
runs = db.get_last_two_runs()

db.close()

run = latest

if run is None:
    st.warning("No evaluation runs found.")
    st.stop()

run_id, timestamp, prompt, model, total, passed, failed, accuracy = run

st.subheader("Latest Evaluation")

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

history = pd.DataFrame(
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

history = pd.DataFrame(
    runs,
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
    history,
    use_container_width=True,
    hide_index=True,
)
st.write("---")
st.subheader("📈 Accuracy Trend")

chart_df = history.copy()

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

if len(runs) >= 2:
    report = RegressionDetector.compare(
        previous_run=runs[1],
        current_run=runs[0],
    )

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