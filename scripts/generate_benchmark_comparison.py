import argparse
import sys
from pathlib import Path

from app.reporting.comparison import BenchmarkComparisonEngine


def main():
    parser = argparse.ArgumentParser(
        description="PromptOps Phase 6.3 Comprehensive Multi-Model Benchmark Comparison Engine"
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("data/validation.db"),
        help="Path to validation SQLite database containing benchmark runs",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("reports/benchmarks/v2/benchmark_manifest.json"),
        help="Path to benchmark manifest JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/benchmarks/v2/comparison"),
        help="Target output directory for comparison artifacts",
    )
    args = parser.parse_args()

    print("=" * 85)
    print("🔬 PROMPTOPS PHASE 6.3: COMPREHENSIVE MULTI-MODEL COMPARATIVE ANALYSIS")
    print("=" * 85)
    print(f"Database Path : {args.db_path}")
    print(f"Manifest Path : {args.manifest_path}")
    print(f"Output Dir    : {args.output_dir}")
    print("-" * 85)

    engine = BenchmarkComparisonEngine(
        db_path=args.db_path,
        manifest_path=args.manifest_path,
        output_dir=args.output_dir,
    )

    try:
        audit_res = engine.load_and_audit()
        print(f"✓ Audited {audit_res['total_models']} models across {audit_res['total_runs']} runs ({audit_res['total_cases']} measured cases). Parity verified.")
    except Exception as e:
        print(f"❌ Audit Error: {e}")
        sys.exit(1)

    print("\n📊 Computing comparative metrics, confusion matrices, and consistency analysis...")
    artifacts = engine.generate_all_artifacts()

    print("\n✅ All 9 Comparative Artifacts Successfully Generated:")
    for name, path in artifacts.items():
        print(f"  • {name:<30} ➔ {path}")

    # Terminal Summary Table
    metrics = engine.compute_comparative_metrics()
    print("\n" + "=" * 90)
    print("🏆 EXECUTIVE MODEL COMPARISON SUMMARY")
    print("=" * 90)
    print(f"{'Model':<18} {'Comp Acc':<10} {'Cat Acc':<10} {'Pri Acc':<10} {'Cat Macro F1':<14} {'Mean Latency':<16} {'Success'}")
    print("-" * 90)
    for m_name, m_info in metrics["models"].items():
        comp_acc = f"{m_info['composite_accuracy']:.2f}%"
        cat_acc = f"{m_info['category_accuracy']:.2f}%"
        pri_acc = f"{m_info['priority_accuracy']:.2f}%"
        macro_f1 = f"{m_info['category_metrics']['macro_avg']['f1']:.4f}"
        lat = f"{m_info['aggregate_latency']['mean_ms']:,.1f} ms"
        succ = f"{m_info['reliability']['success_rate_pct']:.1f}%"
        print(f"{m_name:<18} {comp_acc:<10} {cat_acc:<10} {pri_acc:<10} {macro_f1:<14} {lat:<16} {succ}")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
