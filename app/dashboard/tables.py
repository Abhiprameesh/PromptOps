from typing import Any, List

import pandas as pd
import streamlit as st


def render_history_table(history: List[Any]):
    """
    Renders the overall evaluation runs history dataframe.
    """
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
    return history_df


def render_run_comparison_table(
    run: Any,
    comparison_run: Any,
    selected_run: int,
    compare_run: int,
):
    """
    Renders a table comparing metrics of the two selected runs.
    """
    if comparison_run and run:
        comparison_df = pd.DataFrame(
            {
                "Metric": [
                    "Accuracy",
                    "Passed",
                    "Failed",
                ],
                f"Run {compare_run}": [
                    f"{comparison_run[7]:.2f}%",
                    comparison_run[5],
                    comparison_run[6],
                ],
                f"Run {selected_run}": [
                    f"{run[7]:.2f}%",
                    run[5],
                    run[6],
                ],
            }
        )
        st.table(comparison_df)
    else:
        st.info("Select a run to compare with.")


def render_case_comparison_table(
    case_results: List[Any],
    comparison_case_results: List[Any],
    selected_run: int,
    compare_run: int,
):
    """
    Renders case-by-case verification status showing if cases fixed, regressed, or remained unchanged.
    """
    comparison_rows = []
    comparison_lookup = {row[0]: row for row in comparison_case_results}

    for current_case in case_results:
        case_id = current_case[0]
        previous_case = comparison_lookup.get(case_id)

        if previous_case is None:
            continue

        previous_status = "✅ PASS" if previous_case[3] else "❌ FAIL"
        current_status = "✅ PASS" if current_case[3] else "❌ FAIL"

        if previous_case[3] == current_case[3]:
            change = "No Change"
        elif not previous_case[3] and current_case[3]:
            change = "🟢 Fixed"
        else:
            change = "🔴 Regressed"

        comparison_rows.append(
            {
                "Case ID": case_id,
                f"Run {compare_run}": previous_status,
                f"Run {selected_run}": current_status,
                "Change": change,
            }
        )

    if comparison_rows:
        case_comparison_df = pd.DataFrame(comparison_rows)
        st.dataframe(
            case_comparison_df,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No common test cases to compare.")


def render_case_explorer_table(case_results: List[Any]):
    """
    Renders case outputs with filterable columns and summarizes status metrics.
    """
    if case_results:
        cases_df = pd.DataFrame(
            case_results,
            columns=[
                "Case ID",
                "Expected",
                "Predicted",
                "Passed",
                "Error",
            ],
        )

        cases_df["Status"] = cases_df["Passed"].apply(
            lambda x: "✅ PASS" if x else "❌ FAIL"
        )

        passed = cases_df["Passed"].sum()
        failed = len(cases_df) - passed

        c1, c2, c3 = st.columns(3)
        c1.metric("✅ Passed", passed)
        c2.metric("❌ Failed", failed)
        c3.metric("Success Rate", f"{(passed / len(cases_df)) * 100:.0f}%")

        if cases_df["Error"].notna().any():
            display_df = cases_df[
                ["Case ID", "Expected", "Predicted", "Status", "Error"]
            ]
        else:
            display_df = cases_df[
                ["Case ID", "Expected", "Predicted", "Status"]
            ]

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No case results available.")
