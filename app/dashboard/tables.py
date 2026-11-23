from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st


def render_history_table(history: List[Any]) -> pd.DataFrame:
    """
    Renders the overall evaluation runs history dataframe.
    Compatible with legacy positional tuples.
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


def prepare_classification_table(metrics: Optional[Dict[str, Any]]) -> Optional[pd.DataFrame]:
    """
    Prepares a pandas DataFrame with per-class precision, recall, F1, and support.
    Appends macro and weighted average rows if present.
    """
    if not metrics or not isinstance(metrics, dict):
        return None

    per_class = metrics.get("per_class", {})
    if not per_class or not isinstance(per_class, dict):
        return None

    rows = []
    for cls_name, info in per_class.items():
        if isinstance(info, dict):
            prec = info.get("precision")
            rec = info.get("recall")
            f1 = info.get("f1")
            supp = info.get("support", 0)

            rows.append(
                {
                    "Class": str(cls_name),
                    "Precision": f"{prec:.4f}" if prec is not None else "N/A",
                    "Recall": f"{rec:.4f}" if rec is not None else "N/A",
                    "F1 Score": f"{f1:.4f}" if f1 is not None else "N/A",
                    "Support": supp,
                }
            )

    if not rows:
        return None

    # Macro average row
    macro = metrics.get("macro_avg")
    if isinstance(macro, dict):
        m_prec = macro.get("precision")
        m_rec = macro.get("recall")
        m_f1 = macro.get("f1")
        total_supp = sum(r["Support"] for r in rows)
        rows.append(
            {
                "Class": "📊 Macro Avg",
                "Precision": f"{m_prec:.4f}" if m_prec is not None else "N/A",
                "Recall": f"{m_rec:.4f}" if m_rec is not None else "N/A",
                "F1 Score": f"{m_f1:.4f}" if m_f1 is not None else "N/A",
                "Support": total_supp,
            }
        )

    # Weighted average row
    weighted = metrics.get("weighted_avg")
    if isinstance(weighted, dict):
        w_prec = weighted.get("precision")
        w_rec = weighted.get("recall")
        w_f1 = weighted.get("f1")
        total_supp = sum(r["Support"] for r in rows if not r["Class"].startswith("📊"))
        rows.append(
            {
                "Class": "⚖️ Weighted Avg",
                "Precision": f"{w_prec:.4f}" if w_prec is not None else "N/A",
                "Recall": f"{w_rec:.4f}" if w_rec is not None else "N/A",
                "F1 Score": f"{w_f1:.4f}" if w_f1 is not None else "N/A",
                "Support": total_supp,
            }
        )

    return pd.DataFrame(rows)


def render_classification_table(metrics: Optional[Dict[str, Any]]):
    """
    Renders classification metrics table in Streamlit.
    """
    df = prepare_classification_table(metrics)
    if df is not None and not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Classification metrics are unavailable for this run.")


def prepare_confusion_matrix_df(metrics: Optional[Dict[str, Any]]) -> Optional[pd.DataFrame]:
    """
    Extracts confusion matrix from metrics and formats as a DataFrame.
    Rows = True Classes, Columns = Predicted Classes.
    """
    if not metrics or not isinstance(metrics, dict):
        return None

    cm = metrics.get("confusion_matrix")
    if not cm or not isinstance(cm, dict):
        return None

    # Rows are true classes, cols are pred classes
    cm_df = pd.DataFrame.from_dict(cm, orient="index")
    cm_df = cm_df.fillna(0).astype(int)
    cm_df.index.name = "True Class"
    return cm_df


def prepare_breakdown_tables(
    difficulty_breakdown: Optional[Dict[str, Any]],
    tag_breakdown: Optional[Dict[str, Any]],
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Prepares DataFrames for difficulty and tag evaluation breakdowns.
    """
    diff_df = None
    if difficulty_breakdown and isinstance(difficulty_breakdown, dict):
        diff_rows = []
        for diff_name, info in difficulty_breakdown.items():
            if isinstance(info, dict):
                total = info.get("total", 0)
                passed = info.get("passed", 0)
                failed = total - passed
                acc = info.get("accuracy")
                acc_str = f"{acc:.2f}%" if acc is not None else "N/A"
                diff_rows.append(
                    {
                        "Difficulty Level": str(diff_name).capitalize(),
                        "Total Cases": total,
                        "Passed": passed,
                        "Failed": failed,
                        "Accuracy (%)": acc_str,
                    }
                )
        if diff_rows:
            diff_df = pd.DataFrame(diff_rows)

    tag_df = None
    if tag_breakdown and isinstance(tag_breakdown, dict):
        tag_rows = []
        for tag_name, info in tag_breakdown.items():
            if isinstance(info, dict):
                total = info.get("total", 0)
                passed = info.get("passed", 0)
                failed = total - passed
                acc = info.get("accuracy")
                acc_str = f"{acc:.2f}%" if acc is not None else "N/A"
                tag_rows.append(
                    {
                        "Tag": str(tag_name),
                        "Total Cases": total,
                        "Passed": passed,
                        "Failed": failed,
                        "Accuracy (%)": acc_str,
                    }
                )
        if tag_rows:
            tag_df = pd.DataFrame(tag_rows)

    return diff_df, tag_df


def render_run_comparison_table(
    run: Any,
    comparison_run: Any,
    selected_run: int,
    compare_run: int,
):
    """
    Renders a table comparing metrics of the two selected runs.
    Gracefully handles V1 (8 elements) and V2 (16 elements) runs.
    """
    if comparison_run and run:

        def _val(r, idx, fmt="{:.2f}%"):
            if len(r) > idx and r[idx] is not None:
                try:
                    return fmt.format(r[idx])
                except Exception:
                    return str(r[idx])
            return "N/A"

        metrics_list = [
            ("Overall Accuracy", _val(comparison_run, 7), _val(run, 7)),
            ("Category Accuracy", _val(comparison_run, 8), _val(run, 8)),
            ("Priority Accuracy", _val(comparison_run, 9), _val(run, 9)),
            ("Composite Accuracy", _val(comparison_run, 10), _val(run, 10)),
            ("Avg Latency", _val(comparison_run, 11, "{:.2f} ms"), _val(run, 11, "{:.2f} ms")),
            ("Total Cases", str(comparison_run[4]), str(run[4])),
            ("Passed Cases", str(comparison_run[5]), str(run[5])),
            ("Failed Cases", str(comparison_run[6]), str(run[6])),
        ]

        comparison_df = pd.DataFrame(
            {
                "Metric": [m[0] for m in metrics_list],
                f"Baseline (Run {compare_run})": [m[1] for m in metrics_list],
                f"Current (Run {selected_run})": [m[2] for m in metrics_list],
            }
        )
        st.table(comparison_df)
    else:
        st.info("Select a run to compare with.")


def prepare_case_comparison_data(
    case_results: List[Any],
    comparison_case_results: List[Any],
    selected_run: int,
    compare_run: int,
) -> List[Dict[str, Any]]:
    """
    Extracts case-by-case comparison rows for common test cases.
    Compatible with 5-element (V1) and 12-element (V2) case tuples.
    """
    comparison_rows = []
    comparison_lookup = {row[0]: row for row in comparison_case_results}

    for current_case in case_results:
        case_id = current_case[0]
        previous_case = comparison_lookup.get(case_id)

        if previous_case is None:
            continue

        prev_passed = previous_case[3]
        curr_passed = current_case[3]

        if prev_passed is None or curr_passed is None:
            prev_status = "⚠️ UNKNOWN" if prev_passed is None else ("✅ PASS" if prev_passed else "❌ FAIL")
            curr_status = "⚠️ UNKNOWN" if curr_passed is None else ("✅ PASS" if curr_passed else "❌ FAIL")
            change = "⚠️ Missing Info"
        else:
            prev_status = "✅ PASS" if prev_passed else "❌ FAIL"
            curr_status = "✅ PASS" if curr_passed else "❌ FAIL"

            if bool(prev_passed) == bool(curr_passed):
                change = "⚪ No Change"
            elif not bool(prev_passed) and bool(curr_passed):
                change = "🟢 Fixed"
            else:
                change = "🔴 Regressed"

        # Check for category / priority info
        exp_cat = current_case[1] if len(current_case) > 1 else "N/A"
        exp_pri = current_case[5] if len(current_case) > 5 and current_case[5] is not None else "N/A"

        row_dict = {
            "Case ID": case_id,
            "Category": exp_cat,
            "Priority": exp_pri,
            f"Run {compare_run}": prev_status,
            f"Run {selected_run}": curr_status,
            "Change": change,
        }
        comparison_rows.append(row_dict)

    return comparison_rows


def render_case_comparison_table(
    case_results: List[Any],
    comparison_case_results: List[Any],
    selected_run: int,
    compare_run: int,
):
    """
    Renders case-by-case verification status table.
    """
    rows = prepare_case_comparison_data(
        case_results, comparison_case_results, selected_run, compare_run
    )
    if rows:
        case_comparison_df = pd.DataFrame(rows)
        st.dataframe(
            case_comparison_df,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No common test cases to compare.")


def prepare_case_explorer_data(case_results: List[Any]) -> pd.DataFrame:
    """
    Transforms raw case results into a structured explorer DataFrame.
    Handles both 5-element (legacy) and 12-element (V2) case tuples.
    """
    if not case_results:
        return pd.DataFrame()

    rows = []
    is_v2 = len(case_results[0]) >= 12

    for row in case_results:
        case_id = row[0]
        exp_cat = row[1]
        pred_cat = row[2]
        passed = row[3]
        error = row[4]

        status = "✅ PASS" if passed else ("⚠️ UNKNOWN" if passed is None else "❌ FAIL")

        if is_v2:
            exp_pri = row[5] if row[5] is not None else "N/A"
            pred_pri = row[6] if row[6] is not None else "N/A"
            cat_passed = "✅" if row[7] == 1 else ("❌" if row[7] == 0 else "N/A")
            pri_passed = "✅" if row[8] == 1 else ("❌" if row[8] == 0 else "N/A")
            sum_valid = "✅" if row[9] == 1 else ("❌" if row[9] == 0 else "N/A")
            lat = f"{row[10]:.1f} ms" if row[10] is not None else "N/A"

            rows.append(
                {
                    "Case ID": case_id,
                    "Exp Cat": exp_cat,
                    "Pred Cat": pred_cat,
                    "Cat Pass": cat_passed,
                    "Exp Pri": exp_pri,
                    "Pred Pri": pred_pri,
                    "Pri Pass": pri_passed,
                    "Summary": sum_valid,
                    "Latency": lat,
                    "Overall": status,
                    "Error": error or "",
                }
            )
        else:
            rows.append(
                {
                    "Case ID": case_id,
                    "Expected": exp_cat,
                    "Predicted": pred_cat,
                    "Status": status,
                    "Error": error or "",
                }
            )

    return pd.DataFrame(rows)


def render_case_explorer_table(case_results: List[Any]):
    """
    Renders filterable case output table and summary metrics.
    """
    if not case_results:
        st.info("No case results available.")
        return

    cases_df = prepare_case_explorer_data(case_results)
    if cases_df.empty:
        st.info("No case results available.")
        return

    # Status column name depending on V1 vs V2
    status_col = "Overall" if "Overall" in cases_df.columns else "Status"

    total = len(cases_df)
    passed_count = (cases_df[status_col] == "✅ PASS").sum()
    failed_count = total - passed_count
    rate_str = f"{(passed_count / total) * 100:.1f}%" if total > 0 else "0.0%"

    c1, c2, c3 = st.columns(3)
    c1.metric("✅ Passed Cases", passed_count)
    c2.metric("❌ Failed Cases", failed_count)
    c3.metric("Composite Pass Rate", rate_str)

    # Clean up empty error column if none exist
    if "Error" in cases_df.columns and (cases_df["Error"] == "").all():
        cases_df = cases_df.drop(columns=["Error"])

    st.dataframe(
        cases_df,
        use_container_width=True,
        hide_index=True,
    )
