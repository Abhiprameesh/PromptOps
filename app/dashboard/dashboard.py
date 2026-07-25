import sys
from pathlib import Path
import pandas as pd

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

db = Database()

run = db.get_latest_run()

db.close()

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

db = Database()

runs = db.get_all_runs()

db.close()

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