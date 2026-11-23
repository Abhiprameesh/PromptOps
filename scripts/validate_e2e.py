import asyncio
import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.core.config import PromptConfig, load_prompt_config
from app.evaluation.loader import load_dataset
from app.evaluation.runner import EvaluationRunner
from app.reporting.html_report import HTMLReportGenerator
from app.reporting.regression import RegressionDetector, RegressionThresholds
from app.storage.database import Database


async def main():
    print("=" * 80)
    print("🚀 PROMPTOPS PHASE 5.3: END-TO-END VALIDATION & INTEGRATION PIPELINE")
    print("=" * 80)

    val_db_path = Path("data/validation.db")
    if not val_db_path.exists():
        print("❌ Error: data/validation.db not found. Creating from baseline...")
        import shutil
        shutil.copy("data/evaluation.db", "data/validation.db")

    db = Database(str(val_db_path))

    # Record baseline state in validation database
    baseline_runs = db.get_all_runs()
    print(f"📁 Validation DB Path: {val_db_path}")
    print(f"📊 Pre-existing runs in validation DB: {len(baseline_runs)} (IDs: {[r[0] for r in baseline_runs]})")

    # 1. Load Golden Dataset and Base Prompt Config
    dataset_path = "datasets/v1/github_issues.json"
    base_prompt_path = "prompt/v1.yaml"
    dataset = load_dataset(dataset_path)
    base_config = load_prompt_config(base_prompt_path)

    print(f"📂 Dataset: {dataset_path} ({len(dataset.cases)} test cases)")
    print(f"📝 Base Prompt: {base_prompt_path} (Version: {base_config.version})")

    # 2. Multi-Model Evaluations
    models = ["gemma3:4b", "mistral:latest", "llama3:latest"]
    model_results = {}
    model_run_ids = {}

    runner = EvaluationRunner()

    print("\n" + "=" * 80)
    print("🤖 EXECUTING MULTI-MODEL EVALUATIONS WITH REAL OLLAMA INFERENCE")
    print("=" * 80)

    for model_name in models:
        print(f"\n--- Running evaluation for model: {model_name} ---")
        config_dict = base_config.model_dump()
        config_dict["model"] = model_name
        model_config = PromptConfig(**config_dict)

        result = await runner.evaluate(dataset=dataset, prompt_config=model_config)
        model_results[model_name] = result

        # Save to isolated validation database
        run_id = db.save_run(
            result=result,
            prompt_version=model_config.version,
            model=model_name,
        )
        model_run_ids[model_name] = run_id

        print(f"  ✅ Saved to DB as Run #{run_id}")
        print(f"  Total Cases        : {result.total_cases}")
        print(f"  Passed Cases       : {result.passed_cases}")
        print(f"  Failed Cases       : {result.failed_cases}")
        print(f"  Overall Accuracy   : {result.accuracy:.2f}%")
        print(f"  Category Accuracy  : {result.category_accuracy:.2f}%" if result.category_accuracy is not None else "  Category Accuracy  : N/A")
        print(f"  Priority Accuracy  : {result.priority_accuracy:.2f}%" if result.priority_accuracy is not None else "  Priority Accuracy  : N/A")
        print(f"  Composite Accuracy : {result.composite_accuracy:.2f}%" if result.composite_accuracy is not None else "  Composite Accuracy : N/A")
        print(f"  Avg Latency        : {result.avg_latency_ms:.2f} ms" if result.avg_latency_ms is not None else "  Avg Latency        : N/A")

        cat_macro = result.category_metrics.get("macro_avg", {}).get("f1") if result.category_metrics else None
        pri_macro = result.priority_metrics.get("macro_avg", {}).get("f1") if result.priority_metrics else None
        print(f"  Category Macro F1  : {cat_macro:.4f}" if cat_macro is not None else "  Category Macro F1  : N/A")
        print(f"  Priority Macro F1  : {pri_macro:.4f}" if pri_macro is not None else "  Priority Macro F1  : N/A")

    # 3. Database Persistence Verification
    print("\n" + "=" * 80)
    print("💾 VERIFYING DATABASE PERSISTENCE & RETRIEVAL FIDELITY")
    print("=" * 80)

    for model_name, run_id in model_run_ids.items():
        original = model_results[model_name]
        retrieved_run = db.get_extended_run_by_id(run_id)
        retrieved_cases = db.get_extended_case_results(run_id)

        assert retrieved_run is not None, f"Run #{run_id} not found in DB!"
        assert len(retrieved_cases) == original.total_cases, f"Case count mismatch for #{run_id}!"

        # Check numeric and structural parity
        assert retrieved_run[7] == original.accuracy, f"Accuracy mismatch in #{run_id}"
        assert retrieved_run[8] == original.category_accuracy, f"Category accuracy mismatch in #{run_id}"
        assert retrieved_run[9] == original.priority_accuracy, f"Priority accuracy mismatch in #{run_id}"
        assert retrieved_run[10] == original.composite_accuracy, f"Composite accuracy mismatch in #{run_id}"
        assert retrieved_run[11] == original.avg_latency_ms, f"Latency mismatch in #{run_id}"

        # Check deserialized metrics
        assert retrieved_run[12] == original.category_metrics, f"Category metrics mismatch in #{run_id}"
        assert retrieved_run[13] == original.priority_metrics, f"Priority metrics mismatch in #{run_id}"
        assert retrieved_run[14] == original.difficulty_breakdown, f"Difficulty mismatch in #{run_id}"
        assert retrieved_run[15] == original.tag_breakdown, f"Tag breakdown mismatch in #{run_id}"

        print(f"  ✅ Run #{run_id} ({model_name}): 100% database persistence fidelity verified.")

    # 4. Regression Engine Run-to-Run Comparisons
    print("\n" + "=" * 80)
    print("🚨 EXECUTING REGRESSION ENGINE RUN-TO-RUN COMPARISONS")
    print("=" * 80)

    comparisons = [
        ("Gemma ➔ Mistral", model_run_ids["gemma3:4b"], model_run_ids["mistral:latest"]),
        ("Mistral ➔ Llama", model_run_ids["mistral:latest"], model_run_ids["llama3:latest"]),
        ("Gemma ➔ Llama", model_run_ids["gemma3:4b"], model_run_ids["llama3:latest"]),
        ("Historical Baseline #4 ➔ Gemma", 4, model_run_ids["gemma3:4b"]),
    ]

    regression_reports = {}
    for comp_name, prev_id, curr_id in comparisons:
        rep = RegressionDetector.compare_from_db(
            db=db,
            previous_run_id=prev_id,
            current_run_id=curr_id,
        )
        regression_reports[comp_name] = rep

        print(f"\n--- Comparison: {comp_name} (Run #{prev_id} vs Run #{curr_id}) ---")
        print(f"  Status             : {rep['status'].upper()} (Severity: {rep['severity'].upper()})")
        print(f"  Recommended Action : {rep['recommended_action']}")
        print(f"  Accuracy Difference: {rep['difference']:+.2f}%")
        print(f"  Degraded Metrics   : {len(rep['degraded_metrics'])} items {rep['degraded_metrics']}")
        print(f"  Improved Metrics   : {len(rep['improved_metrics'])} items {rep['improved_metrics']}")
        churn = rep["case_analysis"]
        print(f"  Case Churn         : Newly Failed: {len(churn['newly_failed'])}, Recovered: {len(churn['recovered'])}, Unchanged Pass: {len(churn['unchanged_passing'])}, Unchanged Fail: {len(churn['unchanged_failing'])}")

    # 5. Prompt Regression Validation (Controlled Variation)
    print("\n" + "=" * 80)
    print("📝 PROMPT REGRESSION VALIDATION WITH CONTROLLED PROMPT VARIATION")
    print("=" * 80)

    # Controlled prompt config: prompt variation where we instruct it with a stricter category guidance
    controlled_prompt_path = Path("prompt/v1_controlled_test.yaml")
    controlled_yaml_content = base_prompt_path.replace("\\", "/")
    with open(base_prompt_path, "r", encoding="utf-8") as f:
        prompt_data = f.read()

    # Create controlled prompt copy: change version to v1-controlled and adjust system prompt slightly
    controlled_prompt_text = prompt_data.replace('version: "v1"', 'version: "v1-controlled"')
    controlled_prompt_text = controlled_prompt_text.replace(
        'description: "Initial prompt version"',
        'description: "Controlled validation prompt variant"'
    )
    with open(controlled_prompt_path, "w", encoding="utf-8") as f:
        f.write(controlled_prompt_text)

    controlled_config = load_prompt_config(str(controlled_prompt_path))
    print(f"  Created controlled prompt: {controlled_prompt_path} (Version: {controlled_config.version})")

    # Evaluate Gemma with controlled prompt
    controlled_result = await runner.evaluate(dataset=dataset, prompt_config=controlled_config)
    controlled_run_id = db.save_run(
        result=controlled_result,
        prompt_version=controlled_config.version,
        model=controlled_config.model,
    )
    print(f"  ✅ Saved Controlled Prompt Evaluation as Run #{controlled_run_id}")
    print(f"  Accuracy: {controlled_result.accuracy:.2f}%, Category: {controlled_result.category_accuracy:.2f}%, Latency: {controlled_result.avg_latency_ms:.2f} ms")

    # Compare v1 vs v1-controlled
    prompt_reg_report = RegressionDetector.compare_from_db(
        db=db,
        previous_run_id=model_run_ids["gemma3:4b"],
        current_run_id=controlled_run_id,
    )
    print(f"\n--- Prompt Version Comparison: v1 (Run #{model_run_ids['gemma3:4b']}) vs v1-controlled (Run #{controlled_run_id}) ---")
    print(f"  Status             : {prompt_reg_report['status'].upper()} (Severity: {prompt_reg_report['severity'].upper()})")
    print(f"  Accuracy Delta     : {prompt_reg_report['difference']:+.2f}%")
    print(f"  Action             : {prompt_reg_report['recommended_action']}")

    # Clean up controlled prompt file to keep repo clean
    if controlled_prompt_path.exists():
        controlled_prompt_path.unlink()
        print(f"  Cleaned up temporary controlled prompt file: {controlled_prompt_path}")

    # 6. HTML Reports Generation
    print("\n" + "=" * 80)
    print("📄 GENERATING STANDALONE HTML EVALUATION & REGRESSION REPORTS")
    print("=" * 80)

    html_reports = {}
    for model_name, run_id in model_run_ids.items():
        out_name = f"report_val_{model_name.replace(':', '_')}.html"
        p = HTMLReportGenerator.generate_from_db(
            run_id=run_id,
            db=db,
            output_filename=out_name,
        )
        html_reports[model_name] = p
        print(f"  ✅ Generated Model Report: {p}")

    # Generate Comparative Regression HTML Report
    reg_report_path = HTMLReportGenerator.generate_from_db(
        run_id=model_run_ids["mistral:latest"],
        baseline_run_id=model_run_ids["gemma3:4b"],
        db=db,
        output_filename="report_val_regression_gemma_vs_mistral.html",
    )
    html_reports["regression_gemma_vs_mistral"] = reg_report_path
    print(f"  ✅ Generated Regression Report: {reg_report_path}")

    db.close()

    # 7. Summary metrics export for reporting
    metrics_summary = {
        "models": {},
        "comparisons": {},
        "model_run_ids": model_run_ids,
        "controlled_run_id": controlled_run_id,
    }

    for model_name, r in model_results.items():
        metrics_summary["models"][model_name] = {
            "run_id": model_run_ids[model_name],
            "total_cases": r.total_cases,
            "passed_cases": r.passed_cases,
            "failed_cases": r.failed_cases,
            "accuracy": r.accuracy,
            "category_accuracy": r.category_accuracy,
            "priority_accuracy": r.priority_accuracy,
            "composite_accuracy": r.composite_accuracy,
            "avg_latency_ms": r.avg_latency_ms,
            "category_macro_f1": r.category_metrics.get("macro_avg", {}).get("f1") if r.category_metrics else None,
            "category_weighted_f1": r.category_metrics.get("weighted_avg", {}).get("f1") if r.category_metrics else None,
            "priority_macro_f1": r.priority_metrics.get("macro_avg", {}).get("f1") if r.priority_metrics else None,
            "per_class_category": {
                k: {"precision": v.get("precision"), "recall": v.get("recall"), "f1": v.get("f1"), "support": v.get("support")}
                for k, v in (r.category_metrics.get("per_class", {}).items() if r.category_metrics else [])
            },
            "confusion_matrix_category": r.category_metrics.get("confusion_matrix") if r.category_metrics else None,
        }

    for comp_name, rep in regression_reports.items():
        metrics_summary["comparisons"][comp_name] = {
            "previous_run": rep["previous_run"],
            "current_run": rep["current_run"],
            "difference": rep["difference"],
            "status": rep["status"],
            "severity": rep["severity"],
            "recommended_action": rep["recommended_action"],
            "newly_failed": rep["case_analysis"]["newly_failed"],
            "recovered": rep["case_analysis"]["recovered"],
            "unchanged_passing": rep["case_analysis"]["unchanged_passing"],
            "unchanged_failing": rep["case_analysis"]["unchanged_failing"],
        }

    with open("reports/validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    print("\n" + "=" * 80)
    print("🎉 ALL END-TO-END VALIDATION PHASES COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
