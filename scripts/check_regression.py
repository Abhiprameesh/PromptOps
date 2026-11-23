import argparse
import sys
from typing import Optional

from app.reporting.regression import RegressionDetector, RegressionThresholds
from app.storage.database import Database


def parse_args():
    parser = argparse.ArgumentParser(
        description="PromptOps Automated Regression Detection & CI/CD Gate",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--previous-run",
        type=int,
        default=None,
        help="Baseline run ID (defaults to second latest run)",
    )
    parser.add_argument(
        "--current-run",
        type=int,
        default=None,
        help="Target run ID to evaluate (defaults to latest run)",
    )
    parser.add_argument(
        "--critical-threshold",
        type=float,
        default=None,
        help="Accuracy drop percentage (pp) that triggers a CRITICAL failure (blocks CI)",
    )
    parser.add_argument(
        "--warning-threshold",
        type=float,
        default=None,
        help="Accuracy drop percentage (pp) that triggers a WARNING",
    )
    parser.add_argument(
        "--class-f1-critical",
        type=float,
        default=None,
        help="Per-class F1 drop decimal that triggers a CRITICAL failure",
    )
    parser.add_argument(
        "--class-f1-warning",
        type=float,
        default=None,
        help="Per-class F1 drop decimal that triggers a WARNING",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail (exit code 1) on WARNING severity as well as CRITICAL",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        thresholds = RegressionThresholds.from_env()
        if args.critical_threshold is not None:
            thresholds.accuracy_drop_critical = args.critical_threshold
        if args.warning_threshold is not None:
            thresholds.accuracy_drop_warning = args.warning_threshold
        if args.class_f1_critical is not None:
            thresholds.class_f1_drop_critical = args.class_f1_critical
        if args.class_f1_warning is not None:
            thresholds.class_f1_drop_warning = args.class_f1_warning
        thresholds.validate()
    except ValueError as e:
        print("=" * 75)
        print("❌ CONFIGURATION ERROR")
        print("=" * 75)
        print(f"Configuration error: {str(e)}")
        sys.exit(2)

    db = Database()
    try:
        report = RegressionDetector.compare_from_db(
            db=db,
            previous_run_id=args.previous_run,
            current_run_id=args.current_run,
            thresholds=thresholds,
        )
    except Exception as e:
        print("=" * 75)
        print("❌ REGRESSION DETECTION ERROR")
        print("=" * 75)
        print(f"Error: {str(e)}")
        db.close()
        sys.exit(2)

    db.close()

    # Console Output Formatter
    print("=" * 75)
    print("🤖 PROMPTOPS REGRESSION DETECTION REPORT")
    print("=" * 75)
    print(f"Baseline Run : ID #{report['previous_run']} ({report['previous_timestamp']})")
    print(f"Current Run  : ID #{report['current_run']} ({report['current_timestamp']})")
    print(f"Prompt Ver   : {report['prompt_version']}")
    print(f"Model        : {report['model']}")
    print("-" * 75)

    print("\n📊 KEY METRICS COMPARISON:")
    print(f"{'Metric':<25} {'Baseline':<12} {'Current':<12} {'Delta':<12} {'Status'}")
    print("-" * 75)

    for metric_name, data in report.get("metrics_comparison", {}).items():
        prev_str = f"{data['previous']:.2f}" if data['previous'] is not None else "N/A"
        curr_str = f"{data['current']:.2f}" if data['current'] is not None else "N/A"
        if data['delta'] is not None:
            delta_sign = "+" if data['delta'] > 0 else ""
            delta_str = f"{delta_sign}{data['delta']:.2f}"
        else:
            delta_str = "N/A"

        status_icon = (
            "🔴 CRITICAL"
            if data["status"] == "critical"
            else "🟡 WARNING"
            if data["status"] == "warning"
            else "🟢 IMPROVED"
            if data["status"] == "improved"
            else "⚪ UNCHANGED"
            if data["status"] == "unchanged"
            else "⚪ N/A"
        )

        print(f"{metric_name:<25} {prev_str:<12} {curr_str:<12} {delta_str:<12} {status_icon}")

    # Case Churn
    case_analysis = report.get("case_analysis", {})
    if any(case_analysis.values()):
        print("\n🔍 CASE-LEVEL CHURN:")
        newly_failed = case_analysis.get("newly_failed", [])
        recovered = case_analysis.get("recovered", [])
        unchanged_p = case_analysis.get("unchanged_passing", [])
        unchanged_f = case_analysis.get("unchanged_failing", [])
        added_cases = case_analysis.get("added_cases", [])
        removed_cases = case_analysis.get("removed_cases", [])
        missing_pass_cases = case_analysis.get("missing_pass_cases", [])
        duplicate_cases = case_analysis.get("duplicate_cases", [])

        print(f"  🟢 Recovered (Fixed)    : {len(recovered)} cases {recovered if recovered else ''}")
        print(f"  🔴 Newly Failed (Broke) : {len(newly_failed)} cases {newly_failed if newly_failed else ''}")
        print(f"  ⚪ Unchanged Passing    : {len(unchanged_p)} cases")
        print(f"  ⚪ Unchanged Failing    : {len(unchanged_f)} cases")
        if added_cases:
            print(f"  ➕ Added Cases          : {len(added_cases)} cases {added_cases}")
        if removed_cases:
            print(f"  ➖ Removed Cases        : {len(removed_cases)} cases {removed_cases}")
        if missing_pass_cases:
            print(f"  ⚠️ Missing Status Cases : {len(missing_pass_cases)} cases {missing_pass_cases}")
        if duplicate_cases:
            print(f"  ⚠️ Duplicate Cases      : {len(duplicate_cases)} cases {duplicate_cases}")

    # Class-level F1 changes
    class_analysis = report.get("class_analysis", {})
    degraded_classes = []
    for target in ["category", "priority"]:
        for cls_name, info in class_analysis.get(target, {}).items():
            if info["status"] in ("degraded", "warning", "critical"):
                degraded_classes.append((target, cls_name, info["previous_f1"], info["current_f1"], info["delta"]))

    if degraded_classes:
        print("\n⚠️ DEGRADED CLASS PERFORMANCE:")
        for target, cls_name, p_f1, c_f1, d in degraded_classes:
            print(f"  - [{target.upper()}] '{cls_name}': F1 dropped from {p_f1:.4f} to {c_f1:.4f} ({d:+.4f})")

    # Overall Summary
    print("\n" + "=" * 75)
    severity = report.get("severity", "none").upper()
    status_str = report.get("status", "unchanged").upper()

    if report["severity"] == "critical":
        status_banner = f"🔴 {status_str} (SEVERITY: {severity})"
    elif report["severity"] == "warning":
        status_banner = f"🟡 {status_str} (SEVERITY: {severity})"
    elif report["status"] == "improved":
        status_banner = f"🟢 {status_str} (SEVERITY: {severity})"
    else:
        status_banner = f"⚪ {status_str} (SEVERITY: {severity})"

    print(f"OVERALL STATUS     : {status_banner}")
    print(f"RECOMMENDED ACTION : {report.get('recommended_action')}")
    print("=" * 75)

    # Determine Exit Code for CI/CD
    if report["severity"] == "critical":
        print("\n⛔ CI/CD GATE: FAILED (Critical regression threshold exceeded)")
        sys.exit(1)
    elif report["severity"] == "warning" and args.strict:
        print("\n⛔ CI/CD GATE: FAILED (--strict mode enabled: warning treated as failure)")
        sys.exit(1)
    else:
        print("\n✅ CI/CD GATE: PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
