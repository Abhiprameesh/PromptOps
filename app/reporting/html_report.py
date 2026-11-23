import html as py_html
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from app.reporting.regression import RegressionDetector, RegressionThresholds
from app.storage.database import Database


class HTMLReportGenerator:
    """
    Evaluation Engine V2 HTML Report Generator.
    Generates professional, responsive, standalone HTML evaluation and regression reports.
    Requires no external dependencies, CDNs, or running Streamlit servers.
    """

    REPORT_DIR = Path("reports")

    @classmethod
    def generate(
        cls,
        run_id: int,
        result: Any,
        prompt_version: str,
        model: str,
        regression_report: Optional[Dict[str, Any]] = None,
        output_filename: Optional[str] = None,
    ) -> Path:
        """
        Generates a standalone HTML report from an EvaluationResult object or database record.
        Preserves backward compatibility with existing generate(run_id, result, prompt_version, model).
        """
        cls.REPORT_DIR.mkdir(parents=True, exist_ok=True)

        data = cls._extract_run_data(run_id, result, prompt_version, model)
        html_content = cls._build_html(data, regression_report)

        filename = output_filename or f"report_{run_id}.html"
        output_path = cls.REPORT_DIR / filename

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return output_path

    @classmethod
    def generate_from_db(
        cls,
        run_id: int,
        baseline_run_id: Optional[int] = None,
        db: Optional[Database] = None,
        output_filename: Optional[str] = None,
    ) -> Path:
        """
        Loads run and case records directly from the database, computes regression if
        baseline_run_id is provided, and writes the complete HTML report.
        """
        close_db = False
        if db is None:
            db = Database()
            close_db = True

        try:
            run_data = (
                db.get_extended_run_by_id(run_id)
                if hasattr(db, "get_extended_run_by_id")
                else None
            )
            if run_data is None:
                run_data = db.get_run_by_id(run_id)

            if run_data is None:
                raise ValueError(f"Run #{run_id} not found in database.")

            prompt_version = run_data[2]
            model = run_data[3]

            cases = (
                db.get_extended_case_results(run_id)
                if hasattr(db, "get_extended_case_results")
                else db.get_case_results(run_id)
            )

            regression_report = None
            if baseline_run_id is not None:
                regression_report = RegressionDetector.compare_from_db(
                    db=db,
                    previous_run_id=baseline_run_id,
                    current_run_id=run_id,
                )

            # Package into intermediate data dict
            return cls.generate(
                run_id=run_id,
                result=(run_data, cases),
                prompt_version=prompt_version,
                model=model,
                regression_report=regression_report,
                output_filename=output_filename,
            )
        finally:
            if close_db:
                db.close()

    @classmethod
    def _extract_run_data(
        cls,
        run_id: int,
        result: Any,
        prompt_version: str,
        model: str,
    ) -> Dict[str, Any]:
        """
        Normalizes EvaluationResult Pydantic/dataclass objects, database tuples,
        or legacy structures into a unified dictionary.
        """
        # If passed as (run_tuple, case_tuples) from generate_from_db
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], list):
            run_row, case_rows = result
            return cls._extract_from_db_tuples(run_id, run_row, case_rows, prompt_version, model)

        # If passed an EvaluationResult object (from runner.py / test_runner.py)
        if hasattr(result, "accuracy") and hasattr(result, "case_results"):
            return cls._extract_from_evaluation_result(run_id, result, prompt_version, model)

        # Fallback for raw tuple
        if isinstance(result, (tuple, list)):
            return cls._extract_from_db_tuples(run_id, result, [], prompt_version, model)

        # Minimal fallback
        return {
            "run_id": run_id,
            "prompt_version": prompt_version,
            "model": model,
            "timestamp": "N/A",
            "total_cases": 0,
            "passed_cases": 0,
            "failed_cases": 0,
            "accuracy": None,
            "category_accuracy": None,
            "priority_accuracy": None,
            "composite_accuracy": None,
            "avg_latency_ms": None,
            "category_metrics": None,
            "priority_metrics": None,
            "difficulty_breakdown": None,
            "tag_breakdown": None,
            "case_results": [],
        }

    @classmethod
    def _extract_from_evaluation_result(
        cls,
        run_id: int,
        result: Any,
        prompt_version: str,
        model: str,
    ) -> Dict[str, Any]:
        cases = []
        for c in getattr(result, "case_results", []):
            cases.append(
                {
                    "case_id": getattr(c, "case_id", ""),
                    "expected_category": getattr(c, "expected_category", "N/A"),
                    "predicted_category": getattr(c, "predicted_category", "N/A"),
                    "expected_priority": getattr(c, "expected_priority", None),
                    "predicted_priority": getattr(c, "predicted_priority", None),
                    "passed": getattr(c, "passed", False),
                    "category_passed": getattr(c, "category_passed", None),
                    "priority_passed": getattr(c, "priority_passed", None),
                    "summary_valid": getattr(c, "summary_valid", None),
                    "latency_ms": getattr(c, "latency_ms", None),
                    "error": getattr(c, "error", None),
                    "raw_response": getattr(c, "raw_response", None),
                }
            )

        return {
            "run_id": run_id,
            "prompt_version": prompt_version,
            "model": model,
            "timestamp": "Latest Evaluation",
            "total_cases": getattr(result, "total_cases", len(cases)),
            "passed_cases": getattr(result, "passed_cases", sum(1 for c in cases if c["passed"])),
            "failed_cases": getattr(result, "failed_cases", sum(1 for c in cases if not c["passed"])),
            "accuracy": getattr(result, "accuracy", None),
            "category_accuracy": getattr(result, "category_accuracy", None),
            "priority_accuracy": getattr(result, "priority_accuracy", None),
            "composite_accuracy": getattr(result, "composite_accuracy", None),
            "avg_latency_ms": getattr(result, "avg_latency_ms", None),
            "category_metrics": getattr(result, "category_metrics", None),
            "priority_metrics": getattr(result, "priority_metrics", None),
            "difficulty_breakdown": getattr(result, "difficulty_breakdown", None),
            "tag_breakdown": getattr(result, "tag_breakdown", None),
            "case_results": cases,
        }

    @classmethod
    def _extract_from_db_tuples(
        cls,
        run_id: int,
        run_row: Any,
        case_rows: List[Any],
        prompt_version: str,
        model: str,
    ) -> Dict[str, Any]:
        is_v2 = len(run_row) >= 16

        cases = []
        for cr in case_rows:
            is_case_v2 = len(cr) >= 12
            cases.append(
                {
                    "case_id": cr[0],
                    "expected_category": cr[1],
                    "predicted_category": cr[2],
                    "passed": bool(cr[3]) if cr[3] is not None else None,
                    "error": cr[4],
                    "expected_priority": cr[5] if is_case_v2 else None,
                    "predicted_priority": cr[6] if is_case_v2 else None,
                    "category_passed": bool(cr[7]) if is_case_v2 and cr[7] is not None else None,
                    "priority_passed": bool(cr[8]) if is_case_v2 and cr[8] is not None else None,
                    "summary_valid": bool(cr[9]) if is_case_v2 and cr[9] is not None else None,
                    "latency_ms": cr[10] if is_case_v2 else None,
                    "raw_response": cr[11] if is_case_v2 else None,
                }
            )

        return {
            "run_id": run_row[0] if len(run_row) > 0 else run_id,
            "timestamp": run_row[1] if len(run_row) > 1 else "N/A",
            "prompt_version": run_row[2] if len(run_row) > 2 else prompt_version,
            "model": run_row[3] if len(run_row) > 3 else model,
            "total_cases": run_row[4] if len(run_row) > 4 else len(cases),
            "passed_cases": run_row[5] if len(run_row) > 5 else 0,
            "failed_cases": run_row[6] if len(run_row) > 6 else 0,
            "accuracy": run_row[7] if len(run_row) > 7 else None,
            "category_accuracy": run_row[8] if is_v2 else None,
            "priority_accuracy": run_row[9] if is_v2 else None,
            "composite_accuracy": run_row[10] if is_v2 else None,
            "avg_latency_ms": run_row[11] if is_v2 else None,
            "category_metrics": run_row[12] if is_v2 else None,
            "priority_metrics": run_row[13] if is_v2 else None,
            "difficulty_breakdown": run_row[14] if is_v2 else None,
            "tag_breakdown": run_row[15] if is_v2 else None,
            "case_results": cases,
        }

    @classmethod
    def _build_html(
        cls,
        data: Dict[str, Any],
        regression: Optional[Dict[str, Any]],
    ) -> str:
        """
        Constructs the complete standalone HTML document with embedded CSS.
        """
        css = cls._get_embedded_css()
        overview_html = cls._build_overview_section(data)
        regression_html = cls._build_regression_section(regression)
        classification_html = cls._build_classification_section(data)
        breakdowns_html = cls._build_breakdown_section(data)
        latency_html = cls._build_latency_section(data)
        cases_html = cls._build_case_explorer_section(data)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PromptOps Evaluation Report — Run #{data['run_id']}</title>
<style>
{css}
</style>
</head>
<body>

<div class="report-container">

    <!-- Header Section -->
    <header class="report-header">
        <div class="brand">
            <span class="logo">🤖 PromptOps</span>
            <span class="badge badge-primary">Evaluation Engine V2</span>
        </div>
        <h1 class="report-title">Evaluation & Regression Report</h1>
        <div class="meta-row">
            <span class="meta-item"><b>Run ID:</b> #{data['run_id']}</span>
            <span class="meta-item"><b>Prompt Version:</b> <code>{py_html.escape(str(data['prompt_version']))}</code></span>
            <span class="meta-item"><b>Model:</b> <code>{py_html.escape(str(data['model']))}</code></span>
            <span class="meta-item"><b>Timestamp:</b> {py_html.escape(str(data['timestamp']))}</span>
        </div>
    </header>

    <!-- Main Content -->
    {overview_html}
    {regression_html}
    {classification_html}
    {breakdowns_html}
    {latency_html}
    {cases_html}

    <!-- Footer -->
    <footer class="report-footer">
        <p>Generated by <b>PromptOps Platform</b> • Standalone Technical Audit & CI/CD Report • All assets self-contained</p>
    </footer>

</div>

</body>
</html>
"""

    @classmethod
    def _build_overview_section(cls, data: Dict[str, Any]) -> str:
        def _fmt_pct(val):
            return f"{val:.2f}%" if val is not None else "N/A"

        def _fmt_f1(metrics, key):
            if metrics and isinstance(metrics, dict):
                f1 = metrics.get(key, {}).get("f1")
                if f1 is not None:
                    return f"{f1:.4f}"
            return "N/A"

        def _fmt_lat(val):
            return f"{val:.2f} ms" if val is not None else "N/A"

        acc_str = _fmt_pct(data.get("accuracy"))
        cat_acc_str = _fmt_pct(data.get("category_accuracy"))
        pri_acc_str = _fmt_pct(data.get("priority_accuracy"))
        comp_acc_str = _fmt_pct(data.get("composite_accuracy"))

        cat_macro = _fmt_f1(data.get("category_metrics"), "macro_avg")
        cat_weighted = _fmt_f1(data.get("category_metrics"), "weighted_avg")
        pri_macro = _fmt_f1(data.get("priority_metrics"), "macro_avg")
        lat_str = _fmt_lat(data.get("avg_latency_ms"))

        total = data.get("total_cases", 0)
        passed = data.get("passed_cases", 0)
        failed = data.get("failed_cases", 0)

        return f"""
    <section class="section">
        <h2 class="section-title">📊 Evaluation Overview</h2>
        
        <div class="kpi-grid">
            <div class="kpi-card">
                <span class="kpi-label">Overall Accuracy</span>
                <span class="kpi-value text-accent">{acc_str}</span>
            </div>
            <div class="kpi-card">
                <span class="kpi-label">Category Accuracy</span>
                <span class="kpi-value">{cat_acc_str}</span>
            </div>
            <div class="kpi-card">
                <span class="kpi-label">Priority Accuracy</span>
                <span class="kpi-value">{pri_acc_str}</span>
            </div>
            <div class="kpi-card">
                <span class="kpi-label">Composite Accuracy</span>
                <span class="kpi-value">{comp_acc_str}</span>
            </div>
            <div class="kpi-card">
                <span class="kpi-label">Category Macro F1</span>
                <span class="kpi-value">{cat_macro}</span>
            </div>
            <div class="kpi-card">
                <span class="kpi-label">Category Weighted F1</span>
                <span class="kpi-value">{cat_weighted}</span>
            </div>
            <div class="kpi-card">
                <span class="kpi-label">Priority Macro F1</span>
                <span class="kpi-value">{pri_macro}</span>
            </div>
            <div class="kpi-card">
                <span class="kpi-label">Average Latency</span>
                <span class="kpi-value">{lat_str}</span>
            </div>
        </div>

        <div class="summary-counts-bar">
            <span><b>Total Cases:</b> {total}</span>
            <span class="text-success"><b>Passed:</b> {passed}</span>
            <span class="text-danger"><b>Failed:</b> {failed}</span>
            <span><b>Success Rate:</b> {f"{(passed/total)*100:.1f}%" if total > 0 else "0%"}</span>
        </div>
    </section>
"""

    @classmethod
    def _build_regression_section(cls, regression: Optional[Dict[str, Any]]) -> str:
        if not regression:
            return """
    <section class="section">
        <h2 class="section-title">🚨 Regression Detection & CI/CD Gate</h2>
        <div class="callout callout-info">
            <b>Standalone Run:</b> No baseline evaluation run was provided for automated regression comparison.
        </div>
    </section>
"""

        status = regression.get("status", "unchanged").upper()
        severity = regression.get("severity", "none").upper()
        action = regression.get("recommended_action", "")
        prev_run = regression.get("previous_run")
        curr_run = regression.get("current_run")

        banner_class = "callout-info"
        status_icon = "⚪"
        if severity == "CRITICAL":
            banner_class = "callout-danger"
            status_icon = "🔴"
        elif severity == "WARNING":
            banner_class = "callout-warning"
            status_icon = "🟡"
        elif status == "IMPROVED":
            banner_class = "callout-success"
            status_icon = "🟢"

        # Comparison table rows
        rows_html = ""
        for metric_name, m in regression.get("metrics_comparison", {}).items():
            prev_v = m.get("previous")
            curr_v = m.get("current")
            delta = m.get("delta")
            m_status = m.get("status", "unavailable")

            is_lat = "latency" in metric_name
            fmt = "{:.2f} ms" if is_lat else "{:.2f}%" if "accuracy" in metric_name else "{:.4f}"

            p_str = fmt.format(prev_v) if prev_v is not None else "N/A"
            c_str = fmt.format(curr_v) if curr_v is not None else "N/A"

            if delta is not None:
                sign = "+" if delta > 0 else ""
                d_str = f"{sign}{fmt.format(delta)}"
            else:
                d_str = "N/A"

            badge_css = "badge-neutral"
            if m_status == "critical":
                badge_css = "badge-danger"
            elif m_status == "warning":
                badge_css = "badge-warning"
            elif m_status == "improved":
                badge_css = "badge-success"

            rows_html += f"""
            <tr>
                <td><b>{py_html.escape(metric_name)}</b></td>
                <td>{p_str}</td>
                <td>{c_str}</td>
                <td>{d_str}</td>
                <td><span class="badge {badge_css}">{m_status.upper()}</span></td>
            </tr>
"""

        # Churn badges
        case_analysis = regression.get("case_analysis", {})
        newly_failed = case_analysis.get("newly_failed", [])
        recovered = case_analysis.get("recovered", [])
        unchanged_p = case_analysis.get("unchanged_passing", [])
        unchanged_f = case_analysis.get("unchanged_failing", [])
        added_cases = case_analysis.get("added_cases", [])
        removed_cases = case_analysis.get("removed_cases", [])
        missing_pass = case_analysis.get("missing_pass_cases", [])
        duplicate_cases = case_analysis.get("duplicate_cases", [])

        churn_badges_html = f"""
        <div class="kpi-grid" style="grid-template-columns: repeat(4, 1fr); margin-top: 15px;">
            <div class="kpi-card"><span class="kpi-label">🔴 Newly Failed</span><span class="kpi-value text-danger">{len(newly_failed)}</span></div>
            <div class="kpi-card"><span class="kpi-label">🟢 Recovered</span><span class="kpi-value text-success">{len(recovered)}</span></div>
            <div class="kpi-card"><span class="kpi-label">⚪ Unchanged Passing</span><span class="kpi-value">{len(unchanged_p)}</span></div>
            <div class="kpi-card"><span class="kpi-label">⚪ Unchanged Failing</span><span class="kpi-value">{len(unchanged_f)}</span></div>
            <div class="kpi-card"><span class="kpi-label">➕ Added Cases</span><span class="kpi-value">{len(added_cases)}</span></div>
            <div class="kpi-card"><span class="kpi-label">➖ Removed Cases</span><span class="kpi-value">{len(removed_cases)}</span></div>
            <div class="kpi-card"><span class="kpi-label">⚠️ Missing Pass Info</span><span class="kpi-value">{len(missing_pass)}</span></div>
            <div class="kpi-card"><span class="kpi-label">⚠️ Duplicate IDs</span><span class="kpi-value">{len(duplicate_cases)}</span></div>
        </div>
"""

        # Churn lists details
        churn_details_html = ""
        if newly_failed:
            churn_details_html += f"<p><b>🔴 Newly Failed Cases:</b> <code>{', '.join(newly_failed)}</code></p>"
        if recovered:
            churn_details_html += f"<p><b>🟢 Recovered Cases:</b> <code>{', '.join(recovered)}</code></p>"
        if added_cases:
            churn_details_html += f"<p><b>➕ Added Cases:</b> <code>{', '.join(added_cases)}</code></p>"
        if removed_cases:
            churn_details_html += f"<p><b>➖ Removed Cases:</b> <code>{', '.join(removed_cases)}</code></p>"
        if duplicate_cases:
            churn_details_html += f"<p><b>⚠️ Duplicate Cases:</b> <code>{', '.join(duplicate_cases)}</code></p>"

        # Degraded / Improved callouts
        degraded_metrics = regression.get("degraded_metrics", [])
        improved_metrics = regression.get("improved_metrics", [])
        callouts_html = ""
        if degraded_metrics or improved_metrics:
            callouts_html = "<div class='grid-2' style='margin-top:15px;'>"
            callouts_html += "<div><h4>⚠️ Degraded Metrics & Classes</h4><ul>"
            if degraded_metrics:
                for d in degraded_metrics:
                    callouts_html += f"<li><b>{d.get('metric')}</b>: <code>{d.get('delta')}</code> ({d.get('severity', 'warning').upper()})</li>"
            else:
                callouts_html += "<li>None</li>"
            callouts_html += "</ul></div>"

            callouts_html += "<div><h4>🟢 Improved Metrics & Classes</h4><ul>"
            if improved_metrics:
                for imp in improved_metrics:
                    callouts_html += f"<li><b>{imp.get('metric')}</b>: <code>+{imp.get('delta')}</code></li>"
            else:
                callouts_html += "<li>None</li>"
            callouts_html += "</ul></div></div>"

        return f"""
    <section class="section">
        <h2 class="section-title">🚨 Regression Detection & CI/CD Gate</h2>

        <div class="callout {banner_class}">
            <h3 style="margin-top:0;">{status_icon} Status: {status} (Severity: {severity})</h3>
            <p><b>Recommended Action:</b> {py_html.escape(action)}</p>
            <p style="margin-bottom:0; font-size:12px;">Baseline Run #{prev_run} vs Current Run #{curr_run}</p>
        </div>

        <h3 class="subsection-title">Key Metrics Comparison</h3>
        <table class="report-table">
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Baseline (Run #{prev_run})</th>
                    <th>Current (Run #{curr_run})</th>
                    <th>Delta</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>

        {callouts_html}

        <h3 class="subsection-title">Case-Level Churn Breakdown</h3>
        {churn_badges_html}
        {churn_details_html}
    </section>
"""

    @classmethod
    def _build_classification_section(cls, data: Dict[str, Any]) -> str:
        cat_metrics = data.get("category_metrics")
        pri_metrics = data.get("priority_metrics")

        def _render_classification_target(target_name: str, metrics: Optional[Dict[str, Any]]) -> str:
            if not metrics or not isinstance(metrics, dict) or not metrics.get("per_class"):
                return f"""
            <div class="callout callout-info" style="margin-top:10px;">
                {target_name} classification metrics are unavailable for this run (legacy V1 run).
            </div>
"""
            per_class = metrics.get("per_class", {})
            rows_html = ""
            for cls_name, info in per_class.items():
                p = f"{info.get('precision', 0):.4f}" if info.get('precision') is not None else "N/A"
                r = f"{info.get('recall', 0):.4f}" if info.get('recall') is not None else "N/A"
                f1 = f"{info.get('f1', 0):.4f}" if info.get('f1') is not None else "N/A"
                s = info.get("support", 0)
                rows_html += f"""
                <tr>
                    <td><code>{py_html.escape(str(cls_name))}</code></td>
                    <td>{p}</td>
                    <td>{r}</td>
                    <td><b>{f1}</b></td>
                    <td>{s}</td>
                </tr>
"""

            # Macro & Weighted avg rows
            macro = metrics.get("macro_avg", {})
            weighted = metrics.get("weighted_avg", {})
            total_supp = sum(c.get("support", 0) for c in per_class.values())

            m_p = f"{macro.get('precision', 0):.4f}" if macro.get("precision") is not None else "N/A"
            m_r = f"{macro.get('recall', 0):.4f}" if macro.get("recall") is not None else "N/A"
            m_f1 = f"{macro.get('f1', 0):.4f}" if macro.get("f1") is not None else "N/A"

            w_p = f"{weighted.get('precision', 0):.4f}" if weighted.get("precision") is not None else "N/A"
            w_r = f"{weighted.get('recall', 0):.4f}" if weighted.get("recall") is not None else "N/A"
            w_f1 = f"{weighted.get('f1', 0):.4f}" if weighted.get("f1") is not None else "N/A"

            rows_html += f"""
                <tr class="table-summary-row">
                    <td><b>📊 Macro Average</b></td>
                    <td>{m_p}</td>
                    <td>{m_r}</td>
                    <td><b>{m_f1}</b></td>
                    <td>{total_supp}</td>
                </tr>
                <tr class="table-summary-row">
                    <td><b>⚖️ Weighted Average</b></td>
                    <td>{w_p}</td>
                    <td>{w_r}</td>
                    <td><b>{w_f1}</b></td>
                    <td>{total_supp}</td>
                </tr>
"""

            # Confusion Matrix Table
            cm = metrics.get("confusion_matrix", {})
            cm_html = ""
            if cm and isinstance(cm, dict):
                true_classes = list(cm.keys())
                pred_classes = sorted(list(set(c for row in cm.values() for c in row.keys())))

                headers = "".join(f"<th>Pred: {py_html.escape(c)}</th>" for c in pred_classes)
                cm_rows = ""
                for t in true_classes:
                    cells = ""
                    for p in pred_classes:
                        count = cm[t].get(p, 0)
                        highlight = "class='cm-match'" if t == p and count > 0 else ""
                        cells += f"<td {highlight}>{count}</td>"
                    cm_rows += f"<tr><td><b>True: {py_html.escape(t)}</b></td>{cells}</tr>"

                cm_html = f"""
                <h4 style="margin-top:20px;">Confusion Matrix</h4>
                <table class="report-table cm-table">
                    <thead><tr><th>True \\ Pred</th>{headers}</tr></thead>
                    <tbody>{cm_rows}</tbody>
                </table>
"""

            return f"""
            <table class="report-table" style="margin-top:10px;">
                <thead>
                    <tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1 Score</th><th>Support</th></tr>
                </thead>
                <tbody>{rows_html}</tbody>
            </table>
            {cm_html}
"""

        cat_html = _render_classification_target("Category", cat_metrics)
        pri_html = _render_classification_target("Priority", pri_metrics)

        return f"""
    <section class="section">
        <h2 class="section-title">🎯 Classification Performance Analysis</h2>
        <div class="grid-2">
            <div>
                <h3 class="subsection-title">🏷️ Category Performance</h3>
                {cat_html}
            </div>
            <div>
                <h3 class="subsection-title">⚡ Priority Performance</h3>
                {pri_html}
            </div>
        </div>
    </section>
"""

    @classmethod
    def _build_breakdown_section(cls, data: Dict[str, Any]) -> str:
        diff_b = data.get("difficulty_breakdown")
        tag_b = data.get("tag_breakdown")

        def _render_breakdown_table(title: str, col_name: str, breakdown: Optional[Dict[str, Any]]) -> str:
            if not breakdown or not isinstance(breakdown, dict):
                return f"""
                <div class="callout callout-info">
                    {title} breakdown is unavailable for this run.
                </div>
"""
            rows = ""
            for k, info in breakdown.items():
                if isinstance(info, dict):
                    tot = info.get("total", 0)
                    p = info.get("passed", 0)
                    f = tot - p
                    acc = info.get("accuracy")
                    acc_str = f"{acc:.2f}%" if acc is not None else "N/A"
                    pct = acc if acc is not None else 0.0

                    rows += f"""
                    <tr>
                        <td><b>{py_html.escape(str(k)).capitalize()}</b></td>
                        <td>{tot}</td>
                        <td class="text-success">{p}</td>
                        <td class="text-danger">{f}</td>
                        <td>
                            <div class="progress-bar-wrap">
                                <div class="progress-bar-fill" style="width: {pct}%;"></div>
                                <span class="progress-text">{acc_str}</span>
                            </div>
                        </td>
                    </tr>
"""
            return f"""
            <table class="report-table">
                <thead><tr><th>{col_name}</th><th>Total</th><th>Passed</th><th>Failed</th><th>Accuracy</th></tr></thead>
                <tbody>{rows}</tbody>
            </table>
"""

        diff_html = _render_breakdown_table("Difficulty", "Difficulty Level", diff_b)
        tag_html = _render_breakdown_table("Tag", "Tag Name", tag_b)

        return f"""
    <section class="section">
        <h2 class="section-title">🧩 Slices & Metadata Breakdowns</h2>
        <div class="grid-2">
            <div>
                <h3 class="subsection-title">Difficulty Breakdown</h3>
                {diff_html}
            </div>
            <div>
                <h3 class="subsection-title">Tag Breakdown</h3>
                {tag_html}
            </div>
        </div>
    </section>
"""

    @classmethod
    def _build_latency_section(cls, data: Dict[str, Any]) -> str:
        cases = data.get("case_results", [])
        latencies = [c["latency_ms"] for c in cases if c.get("latency_ms") is not None]

        if not latencies:
            return """
    <section class="section">
        <h2 class="section-title">⏱️ Latency Analysis</h2>
        <div class="callout callout-info">
            Per-case inference latency is unavailable for this run (legacy V1 run).
        </div>
    </section>
"""

        avg_lat = sum(latencies) / len(latencies)
        min_lat = min(latencies)
        max_lat = max(latencies)

        # Build latency bar list (limit to first 15 or full)
        bars_html = ""
        max_scale = max(max_lat, 1.0)
        for c in cases:
            lat = c.get("latency_ms")
            if lat is not None:
                pct = min(100.0, (lat / max_scale) * 100.0)
                css_class = "bar-success" if c.get("passed") else "bar-danger"
                bars_html += f"""
                <div class="latency-row">
                    <span class="latency-label"><code>{py_html.escape(c['case_id'])}</code></span>
                    <div class="latency-bar-track">
                        <div class="latency-bar-fill {css_class}" style="width: {pct}%;"></div>
                    </div>
                    <span class="latency-value">{lat:.1f} ms</span>
                </div>
"""

        return f"""
    <section class="section">
        <h2 class="section-title">⏱️ Inference Latency Analysis</h2>
        
        <div class="kpi-grid" style="grid-template-columns: repeat(3, 1fr); margin-bottom: 20px;">
            <div class="kpi-card"><span class="kpi-label">Average Latency</span><span class="kpi-value">{avg_lat:.2f} ms</span></div>
            <div class="kpi-card"><span class="kpi-label">Min Latency</span><span class="kpi-value">{min_lat:.2f} ms</span></div>
            <div class="kpi-card"><span class="kpi-label">Max Latency</span><span class="kpi-value">{max_lat:.2f} ms</span></div>
        </div>

        <h3 class="subsection-title">Per-Case Latency Distribution</h3>
        <div class="latency-container">
            {bars_html}
        </div>
    </section>
"""

    @classmethod
    def _build_case_explorer_section(cls, data: Dict[str, Any]) -> str:
        cases = data.get("case_results", [])
        if not cases:
            return """
    <section class="section">
        <h2 class="section-title">📋 Case Explorer</h2>
        <div class="callout callout-info">No test case records available.</div>
    </section>
"""

        # Check if cases have V2 fields
        has_v2 = any(c.get("expected_priority") is not None or c.get("latency_ms") is not None for c in cases)

        rows_html = ""
        for c in cases:
            p = c.get("passed")
            status_text = "PASS" if p else ("UNKNOWN" if p is None else "FAIL")
            status_badge = "badge-success" if p else ("badge-warning" if p is None else "badge-danger")

            err = c.get("error")
            err_str = f"<span class='text-danger'>{py_html.escape(str(err))}</span>" if err else "—"

            if has_v2:
                cat_pass = "✅" if c.get("category_passed") else ("❌" if c.get("category_passed") is False else "—")
                pri_pass = "✅" if c.get("priority_passed") else ("❌" if c.get("priority_passed") is False else "—")
                sum_pass = "✅" if c.get("summary_valid") else ("❌" if c.get("summary_valid") is False else "—")
                lat = f"{c['latency_ms']:.1f} ms" if c.get("latency_ms") is not None else "—"
                exp_pri = py_html.escape(str(c.get("expected_priority") or "—"))
                pred_pri = py_html.escape(str(c.get("predicted_priority") or "—"))

                rows_html += f"""
                <tr>
                    <td><code>{py_html.escape(c['case_id'])}</code></td>
                    <td>{py_html.escape(str(c['expected_category']))}</td>
                    <td>{py_html.escape(str(c['predicted_category']))}</td>
                    <td>{cat_pass}</td>
                    <td>{exp_pri}</td>
                    <td>{pred_pri}</td>
                    <td>{pri_pass}</td>
                    <td>{sum_pass}</td>
                    <td>{lat}</td>
                    <td><span class="badge {status_badge}">{status_text}</span></td>
                    <td>{err_str}</td>
                </tr>
"""
            else:
                rows_html += f"""
                <tr>
                    <td><code>{py_html.escape(c['case_id'])}</code></td>
                    <td>{py_html.escape(str(c['expected_category']))}</td>
                    <td>{py_html.escape(str(c['predicted_category']))}</td>
                    <td><span class="badge {status_badge}">{status_text}</span></td>
                    <td>{err_str}</td>
                </tr>
"""

        header_html = (
            "<tr><th>Case ID</th><th>Exp Cat</th><th>Pred Cat</th><th>Cat</th><th>Exp Pri</th><th>Pred Pri</th><th>Pri</th><th>Sum</th><th>Latency</th><th>Status</th><th>Error</th></tr>"
            if has_v2
            else "<tr><th>Case ID</th><th>Expected</th><th>Predicted</th><th>Status</th><th>Error</th></tr>"
        )

        return f"""
    <section class="section">
        <h2 class="section-title">📋 Case Explorer & Detailed Predictions</h2>
        <table class="report-table">
            <thead>{header_html}</thead>
            <tbody>{rows_html}</tbody>
        </table>
    </section>
"""

    @classmethod
    def _get_embedded_css(cls) -> str:
        """
        Returns modern embedded CSS for complete offline standalone rendering.
        """
        return """
:root {
    --bg-primary: #0f172a;
    --bg-surface: #1e293b;
    --bg-card: #273549;
    --border-color: #334155;
    --text-primary: #f8fafc;
    --text-muted: #94a3b8;
    --accent: #3b82f6;
    --accent-hover: #60a5fa;
    --success: #22c55e;
    --warning: #f59e0b;
    --danger: #ef4444;
}

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background-color: var(--bg-primary);
    color: var(--text-primary);
    line-height: 1.5;
    padding: 24px;
}

.report-container {
    max-width: 1280px;
    margin: 0 auto;
}

/* Header */
.report-header {
    background-color: var(--bg-surface);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 24px;
}

.brand {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 8px;
}

.logo {
    font-size: 20px;
    font-weight: 700;
    letter-spacing: -0.5px;
}

.report-title {
    font-size: 26px;
    font-weight: 800;
    margin-bottom: 16px;
}

.meta-row {
    display: flex;
    flex-wrap: wrap;
    gap: 18px;
    font-size: 14px;
    color: var(--text-muted);
}

.meta-item b {
    color: var(--text-primary);
}

/* Sections */
.section {
    background-color: var(--bg-surface);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 24px;
}

.section-title {
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 18px;
    border-bottom: 1px solid var(--border-color);
    padding-bottom: 10px;
}

.subsection-title {
    font-size: 16px;
    font-weight: 600;
    margin: 16px 0 10px 0;
    color: var(--text-primary);
}

/* KPI Grids */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 14px;
    margin-bottom: 16px;
}

.kpi-card {
    background-color: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 14px;
    display: flex;
    flex-direction: column;
}

.kpi-label {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
}

.kpi-value {
    font-size: 20px;
    font-weight: 700;
}

.summary-counts-bar {
    background-color: var(--bg-card);
    border-radius: 6px;
    padding: 10px 16px;
    display: flex;
    gap: 24px;
    font-size: 14px;
}

/* Grids */
.grid-2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
}

@media (max-width: 900px) {
    .grid-2 {
        grid-template-columns: 1fr;
    }
}

/* Tables */
.report-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    margin-top: 8px;
}

.report-table th, .report-table td {
    padding: 10px 12px;
    border: 1px solid var(--border-color);
    text-align: left;
}

.report-table th {
    background-color: var(--bg-card);
    color: var(--text-muted);
    font-weight: 600;
    text-transform: uppercase;
    font-size: 11px;
    letter-spacing: 0.5px;
}

.report-table tbody tr:hover {
    background-color: rgba(255, 255, 255, 0.02);
}

.table-summary-row {
    background-color: var(--bg-card);
    font-weight: 600;
}

.cm-table td, .cm-table th {
    text-align: center;
}

.cm-match {
    background-color: rgba(34, 197, 94, 0.15);
    font-weight: bold;
    color: var(--success);
}

/* Badges */
.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.badge-primary { background-color: rgba(59, 130, 246, 0.2); color: var(--accent); }
.badge-success { background-color: rgba(34, 197, 94, 0.2); color: var(--success); }
.badge-warning { background-color: rgba(245, 158, 11, 0.2); color: var(--warning); }
.badge-danger  { background-color: rgba(239, 68, 68, 0.2); color: var(--danger); }
.badge-neutral { background-color: rgba(148, 163, 184, 0.2); color: var(--text-muted); }

/* Callouts */
.callout {
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 16px;
    font-size: 14px;
    border-left: 4px solid;
}

.callout-info    { background-color: rgba(59, 130, 246, 0.1); border-color: var(--accent); }
.callout-success { background-color: rgba(34, 197, 94, 0.1); border-color: var(--success); }
.callout-warning { background-color: rgba(245, 158, 11, 0.1); border-color: var(--warning); }
.callout-danger  { background-color: rgba(239, 68, 68, 0.1); border-color: var(--danger); }

/* Progress bar */
.progress-bar-wrap {
    background-color: var(--bg-primary);
    border-radius: 4px;
    height: 18px;
    position: relative;
    overflow: hidden;
    display: flex;
    align-items: center;
}

.progress-bar-fill {
    background-color: var(--accent);
    height: 100%;
    transition: width 0.3s ease;
}

.progress-text {
    position: absolute;
    width: 100%;
    text-align: center;
    font-size: 11px;
    font-weight: 700;
    color: #fff;
    text-shadow: 0 1px 2px rgba(0,0,0,0.8);
}

/* Latency Bar Distribution */
.latency-container {
    max-height: 350px;
    overflow-y: auto;
    padding-right: 8px;
}

.latency-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 6px;
    font-size: 12px;
}

.latency-label {
    width: 90px;
    flex-shrink: 0;
}

.latency-bar-track {
    flex-grow: 1;
    background-color: var(--bg-card);
    height: 12px;
    border-radius: 4px;
    overflow: hidden;
}

.latency-bar-fill {
    height: 100%;
    border-radius: 4px;
}

.bar-success { background-color: var(--success); }
.bar-danger  { background-color: var(--danger); }

.latency-value {
    width: 70px;
    text-align: right;
    font-weight: 600;
}

/* Helpers */
.text-accent  { color: var(--accent); }
.text-success { color: var(--success); }
.text-warning { color: var(--warning); }
.text-danger  { color: var(--danger); }

code {
    background-color: rgba(0,0,0,0.3);
    padding: 2px 6px;
    border-radius: 4px;
    font-family: monospace;
    font-size: 12px;
}

/* Footer */
.report-footer {
    text-align: center;
    font-size: 12px;
    color: var(--text-muted);
    margin-top: 32px;
    padding-top: 16px;
    border-top: 1px solid var(--border-color);
}
"""
