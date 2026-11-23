from typing import Any, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render_accuracy_trend_chart(history_df: pd.DataFrame):
    """
    Renders the Plotly line chart displaying accuracy trends across all runs.
    """
    if history_df.empty:
        st.info("No run history available.")
        return

    chart_df = history_df.copy()

    # Show oldest → newest
    if "Run ID" in chart_df.columns:
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
        yaxis_range=[0, 105],
        template="plotly_dark",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


def render_confusion_matrix_chart(confusion_df: pd.DataFrame, title: str = "Confusion Matrix"):
    """
    Renders an interactive heatmap for the classification confusion matrix.
    Rows = True Classes, Columns = Predicted Classes.
    """
    if confusion_df.empty:
        st.info("No confusion matrix data to display.")
        return

    fig = px.imshow(
        confusion_df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        labels=dict(x="Predicted Class", y="True Class", color="Count"),
        title=title,
    )

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Predicted Class",
        yaxis_title="True Class",
    )

    st.plotly_chart(fig, use_container_width=True)


def render_latency_chart(case_results: List[Any], avg_latency: Optional[float] = None):
    """
    Renders an interactive bar chart of inference latency per test case.
    """
    if not case_results:
        st.info("No case results available for latency analysis.")
        return

    # Extract case_id and latency_ms (index 10 in extended case results)
    latency_data = []
    for row in case_results:
        cid = row[0]
        passed = row[3] if len(row) > 3 else None
        lat = row[10] if len(row) > 10 else None

        if lat is not None:
            status_label = "Pass" if passed else "Fail"
            latency_data.append({"Case ID": str(cid), "Latency (ms)": round(float(lat), 2), "Status": status_label})

    if not latency_data:
        st.info("Per-case latency data is not available for this run (legacy V1 run).")
        return

    lat_df = pd.DataFrame(latency_data)

    fig = px.bar(
        lat_df,
        x="Case ID",
        y="Latency (ms)",
        color="Status",
        color_discrete_map={"Pass": "#2ECC71", "Fail": "#E74C3C"},
        title="Inference Latency per Test Case (ms)",
    )

    if avg_latency is not None and avg_latency > 0:
        fig.add_hline(
            y=avg_latency,
            line_dash="dash",
            line_color="#F39C12",
            annotation_text=f"Avg Latency: {avg_latency:.1f} ms",
            annotation_position="top left",
        )

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Test Case ID",
        yaxis_title="Latency (ms)",
    )

    st.plotly_chart(fig, use_container_width=True)


def render_breakdown_bar_chart(df: pd.DataFrame, category_col: str, title: str):
    """
    Renders a bar chart comparing accuracy across slices (difficulty or tags).
    """
    if df is None or df.empty or category_col not in df.columns:
        return

    chart_df = df.copy()
    if "Accuracy (%)" in chart_df.columns:
        # Convert string percentages like '80.00%' or float
        chart_df["Accuracy_Num"] = (
            chart_df["Accuracy (%)"].astype(str).str.rstrip("%").astype(float)
        )

        fig = px.bar(
            chart_df,
            x=category_col,
            y="Accuracy_Num",
            text="Accuracy (%)",
            title=title,
            color="Accuracy_Num",
            color_continuous_scale="Tealgrn",
        )

        fig.update_layout(
            template="plotly_dark",
            xaxis_title=category_col,
            yaxis_title="Accuracy (%)",
            yaxis_range=[0, 105],
        )

        st.plotly_chart(fig, use_container_width=True)
