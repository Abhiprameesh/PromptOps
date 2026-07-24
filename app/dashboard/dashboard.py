import sys
from pathlib import Path

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

st.write(f"**Prompt Version:** {prompt}")
st.write(f"**Model:** {model}")
st.write(f"**Timestamp:** {timestamp}")