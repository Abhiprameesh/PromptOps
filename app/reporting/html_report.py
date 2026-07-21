from pathlib import Path


class HTMLReportGenerator:

    REPORT_DIR = Path("reports")

    @classmethod
    def generate(cls, run_id, result, prompt_version, model):

        cls.REPORT_DIR.mkdir(exist_ok=True)

        html = f"""
<!DOCTYPE html>
<html>
<head>

<title>PromptOps Report</title>

<style>

body {{
    font-family: Arial;
    margin:40px;
    background:#f5f5f5;
}}

table {{
    border-collapse: collapse;
    width:100%;
}}

th, td {{
    border:1px solid #ccc;
    padding:10px;
}}

th {{
    background:#333;
    color:white;
}}

.pass {{
    color:green;
    font-weight:bold;
}}

.fail {{
    color:red;
    font-weight:bold;
}}

.summary {{
    margin-bottom:25px;
}}

</style>

</head>

<body>

<h1>PromptOps Evaluation Report</h1>

<div class="summary">

<h3>Run ID: {run_id}</h3>

<p><b>Prompt Version:</b> {prompt_version}</p>

<p><b>Model:</b> {model}</p>

<p><b>Accuracy:</b> {result.accuracy:.2f}%</p>

<p><b>Passed:</b> {result.passed_cases}</p>

<p><b>Failed:</b> {result.failed_cases}</p>

</div>

<table>

<tr>
<th>Case ID</th>
<th>Expected</th>
<th>Predicted</th>
<th>Status</th>
</tr>
"""

        for case in result.case_results:

            status = "PASS" if case.passed else "FAIL"
            css = "pass" if case.passed else "fail"

            html += f"""
<tr>
<td>{case.case_id}</td>
<td>{case.expected_category}</td>
<td>{case.predicted_category}</td>
<td class="{css}">{status}</td>
</tr>
"""

        html += """
</table>

</body>
</html>
"""

        output_path = cls.REPORT_DIR / f"report_{run_id}.html"

        with open(output_path, "w", encoding="utf-8") as file:
            file.write(html)

        return output_path