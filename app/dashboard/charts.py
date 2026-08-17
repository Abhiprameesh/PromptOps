import pandas as pd
import plotly.express as px
import streamlit as st


def render_accuracy_trend_chart(history_df: pd.DataFrame):
    """
    Renders the Plotly line chart displaying accuracy trends across all runs.
    """
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
