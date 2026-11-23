import argparse
import asyncio
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import PromptConfig, load_prompt_config
from app.core.schemas import GitHubIssue, IssueClassification
from app.evaluation.results import CaseResult, EvaluationResult
from app.evaluation.runner import EvaluationRunner
from app.evaluation.schema import GoldenDataset, GoldenDatasetCase
from app.llm.formatter import build_user_prompt
from app.llm.parser import parse_llm_response
from app.services.inference import InferenceService
from app.storage.database import Database

BENCHMARK_MODELS = [
    "gemma3:4b",
    "mistral:latest",
    "llama3:latest",
]

DEFAULT_DATASET = Path("datasets/v2/github_issues.json")
DEFAULT_PROMPT = Path("prompt/v1.yaml")
DEFAULT_DB = Path("data/validation.db")
OUTPUT_DIR = Path("reports/benchmarks/v2")
MANIFEST_PATH = OUTPUT_DIR / "benchmark_manifest.json"
RESULTS_PATH = OUTPUT_DIR / "benchmark_results.json"


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def get_ollama_version() -> str:
    """Retrieve local Ollama server version via CLI."""
    try:
        out = subprocess.check_output(["ollama", "--version"], text=True, timeout=5)
        return out.strip()
    except Exception as e:
        return f"Unknown ({e})"


def get_hardware_info() -> Dict[str, Any]:
    """Capture host system hardware and OS environment."""
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "python_version": sys.version,
        "ollama_version": get_ollama_version(),
    }


def load_or_create_manifest() -> Dict[str, Any]:
    """Load existing manifest or initialize a new one."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if MANIFEST_PATH.exists():
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    return {
        "benchmark_version": "v2",
        "created_at": datetime.now().isoformat(),
        "environment": get_hardware_info(),
        "dataset": {
            "path": str(DEFAULT_DATASET),
            "sha256": compute_file_sha256(DEFAULT_DATASET) if DEFAULT_DATASET.exists() else None,
        },
        "prompt": {
            "path": str(DEFAULT_PROMPT),
            "sha256": compute_file_sha256(DEFAULT_PROMPT) if DEFAULT_PROMPT.exists() else None,
        },
        "models": BENCHMARK_MODELS,
        "target_database": str(DEFAULT_DB),
        "repetitions_planned": 3,
        "runs": {},
    }


def save_manifest(manifest: Dict[str, Any]):
    """Persist the manifest to disk safely."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = MANIFEST_PATH.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    temp_path.replace(MANIFEST_PATH)


def load_or_create_results() -> Dict[str, Any]:
    """Load existing benchmark results or initialize a new structure."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if RESULTS_PATH.exists():
        try:
            with open(RESULTS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    return {
        "benchmark_version": "v2",
        "updated_at": datetime.now().isoformat(),
        "models": {},
    }


def save_results(results: Dict[str, Any]):
    """Persist benchmark results to disk safely."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = RESULTS_PATH.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    temp_path.replace(RESULTS_PATH)


async def execute_warmup(model: str, prompt_config: PromptConfig) -> Dict[str, Any]:
    """Execute exactly one warm-up inference (excluded from measured metrics)."""
    service = InferenceService()
    test_issue = GitHubIssue(
        title="Warmup check for model loading",
        body="This is an automated warm-up request to load the model into memory. Classify as bug.",
    )
    model_cfg = prompt_config.model_copy(update={"model": model})

    t0 = time.perf_counter()
    raw_resp = await service.client.generate(
        model_cfg,
        build_user_prompt(test_issue, model_cfg),
    )
    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    parsed = parse_llm_response(raw_resp)

    return {
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "latency_ms": latency_ms,
        "parsed_successfully": parsed is not None,
    }


async def run_smoke_test(
    models: List[str],
    dataset_path: Path,
    prompt_path: Path,
    db_path: Path,
) -> bool:
    """
    Execute smoke test on 1 case per model to verify end-to-end functionality
    prior to launching the full benchmark.
    """
    print("\n" + "=" * 75)
    print("🧪 PROMPTOPS PHASE 6.2 PRE-FLIGHT SMOKE TEST")
    print("=" * 75)

    with open(dataset_path, "r", encoding="utf-8") as f:
        full_data = json.load(f)
    single_case = GoldenDataset.model_validate({
        "version": full_data["version"],
        "description": "Smoke test subset",
        "created_at": full_data["created_at"],
        "cases": [full_data["cases"][0]],  # Only case GH001
    })

    base_prompt_cfg = load_prompt_config(str(prompt_path))
    runner = EvaluationRunner()

    all_passed = True
    for model in models:
        print(f"\n[Smoke Test] Testing model: {model} ...")
        t0 = time.perf_counter()
        model_cfg = base_prompt_cfg.model_copy(update={"model": model})

        try:
            result = await runner.evaluate(single_case, model_cfg)
            elapsed = time.perf_counter() - t0
            case_res = result.case_results[0]

            print(f"  ✓ Inference returned in {elapsed:.2f}s (Case latency: {case_res.latency_ms} ms)")
            print(f"  ✓ Predicted Category: {case_res.predicted_category} | Priority: {case_res.predicted_priority}")
            print(f"  ✓ Passed: {case_res.passed} | Error: {case_res.error}")

            if case_res.error is not None:
                print(f"  ❌ Model {model} produced error during inference: {case_res.error}")
                all_passed = False
            else:
                print(f"  ✅ Smoke test for {model} PASSED")
        except Exception as e:
            print(f"  ❌ Model {model} crashed during smoke test: {e}")
            all_passed = False

    print("\n" + "=" * 75)
    if all_passed:
        print("✅ ALL MODELS PASSED PRE-FLIGHT SMOKE TEST")
    else:
        print("❌ PRE-FLIGHT SMOKE TEST FAILED - RESOLVE ISSUES BEFORE RUNNING BENCHMARK")
    print("=" * 75 + "\n")
    return all_passed


def compute_reliability_and_latency(result: EvaluationResult) -> Dict[str, Any]:
    """Compute detailed latency and reliability statistics across repetition case results."""
    valid_latencies = [
        c.latency_ms for c in result.case_results if c.latency_ms is not None and c.latency_ms > 0
    ]

    if valid_latencies:
        mean_lat = round(sum(valid_latencies) / len(valid_latencies), 2)
        med_lat = round(statistics.median(valid_latencies), 2)
        min_lat = round(min(valid_latencies), 2)
        max_lat = round(max(valid_latencies), 2)
        var_lat = round(statistics.variance(valid_latencies), 2) if len(valid_latencies) > 1 else 0.0
    else:
        mean_lat = med_lat = min_lat = max_lat = var_lat = 0.0

    successful_inferences = sum(1 for c in result.case_results if c.error is None)
    failed_inferences = sum(1 for c in result.case_results if c.error is not None)
    invalid_json_responses = sum(
        1 for c in result.case_results if c.raw_response is not None and c.predicted_category is None
    )
    parsing_failures = sum(
        1 for c in result.case_results if c.error is not None and "JSON" in c.error.upper()
    )
    missing_or_invalid_fields = sum(
        1 for c in result.case_results if c.predicted_category is None or c.predicted_priority is None or not c.summary_valid
    )

    return {
        "latency": {
            "avg_ms": mean_lat,
            "median_ms": med_lat,
            "min_ms": min_lat,
            "max_ms": max_lat,
            "variance": var_lat,
            "valid_samples": len(valid_latencies),
        },
        "reliability": {
            "total_cases": result.total_cases,
            "successful_inferences": successful_inferences,
            "failed_inferences": failed_inferences,
            "invalid_json_responses": invalid_json_responses,
            "parsing_failures": parsing_failures,
            "missing_or_invalid_fields": missing_or_invalid_fields,
        },
    }


async def run_benchmark(
    models: Optional[List[str]] = None,
    specific_rep: Optional[int] = None,
    dataset_path: Path = DEFAULT_DATASET,
    prompt_path: Path = DEFAULT_PROMPT,
    db_path: Path = DEFAULT_DB,
):
    """
    Execute multi-model benchmark with resumability and warm-up isolation.
    """
    target_models = models or BENCHMARK_MODELS
    manifest = load_or_create_manifest()
    results = load_or_create_results()

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset_raw = json.load(f)
    dataset = GoldenDataset.model_validate(dataset_raw)
    base_prompt_cfg = load_prompt_config(str(prompt_path))

    print("\n" + "=" * 80)
    print("🚀 PROMPTOPS PHASE 6.2: MULTI-MODEL BENCHMARK EXECUTION")
    print("=" * 80)
    print(f"Target Models : {', '.join(target_models)}")
    print(f"Dataset       : {dataset_path} ({len(dataset.cases)} cases, SHA-256: {manifest['dataset']['sha256'][:10]}...)")
    print(f"Prompt Config : {prompt_path} (Version: {base_prompt_cfg.version})")
    print(f"Database      : {db_path}")
    print(f"Repetitions   : 3 per model (Sequential execution)")
    print("=" * 80)

    runner = EvaluationRunner()

    for model in target_models:
        model_slug = model.replace(":", "_").replace("/", "_")
        print(f"\n================================================================================")
        print(f"📌 MODEL: {model}")
        print(f"================================================================================")

        # 1. Warm-up phase
        warmup_key = f"{model_slug}_warmup"
        if warmup_key not in manifest.get("warmups", {}):
            print(f"🔥 Executing model warm-up for {model} (excluded from benchmark metrics)...")
            try:
                warmup_info = await execute_warmup(model, base_prompt_cfg)
                if "warmups" not in manifest:
                    manifest["warmups"] = {}
                manifest["warmups"][warmup_key] = warmup_info
                save_manifest(manifest)
                print(f"  ✓ Warm-up completed in {warmup_info['latency_ms']} ms")
            except Exception as e:
                print(f"  ❌ Warm-up failed for {model}: {e}")
                manifest["warmups"][warmup_key] = {"error": str(e), "timestamp": datetime.now().isoformat()}
                save_manifest(manifest)
                continue
        else:
            print(f"  ✓ Model warm-up already recorded for {model}")

        # 2. Measured Repetitions
        reps_to_run = [specific_rep] if specific_rep is not None else [1, 2, 3]

        for rep in reps_to_run:
            run_key = f"{model_slug}_rep_{rep}"
            existing_run_info = manifest["runs"].get(run_key)

            # Resumability check
            if existing_run_info and existing_run_info.get("status") == "completed":
                print(f"\n⏩ [Repetition {rep}/3] Already completed (Run ID: #{existing_run_info.get('run_id')}). Skipping.")
                continue

            print(f"\n▶️ [Repetition {rep}/3] Starting evaluation over {len(dataset.cases)} cases...")
            model_cfg = base_prompt_cfg.model_copy(update={"model": model})

            # Record run start in manifest
            manifest["runs"][run_key] = {
                "model": model,
                "repetition": rep,
                "status": "in_progress",
                "started_at": datetime.now().isoformat(),
            }
            save_manifest(manifest)

            rep_start_time = time.perf_counter()
            try:
                eval_result = await runner.evaluate(dataset, model_cfg)
                duration_sec = round(time.perf_counter() - rep_start_time, 2)

                # Save immediately to validation DB
                with Database(db_path) as db:
                    run_id = db.save_run(eval_result, model_cfg.version, model)

                stats = compute_reliability_and_latency(eval_result)

                # Record completion in manifest
                manifest["runs"][run_key] = {
                    "model": model,
                    "repetition": rep,
                    "run_id": run_id,
                    "status": "completed",
                    "started_at": manifest["runs"][run_key]["started_at"],
                    "completed_at": datetime.now().isoformat(),
                    "duration_sec": duration_sec,
                    "accuracy": eval_result.accuracy,
                    "category_accuracy": eval_result.category_accuracy,
                    "priority_accuracy": eval_result.priority_accuracy,
                    "composite_accuracy": eval_result.composite_accuracy,
                    "avg_latency_ms": eval_result.avg_latency_ms,
                    "category_macro_f1": eval_result.category_metrics.get("macro_avg", {}).get("f1") if eval_result.category_metrics else None,
                    "priority_macro_f1": eval_result.priority_metrics.get("macro_avg", {}).get("f1") if eval_result.priority_metrics else None,
                    "stats": stats,
                }
                save_manifest(manifest)

                # Update benchmark_results.json
                if model not in results["models"]:
                    results["models"][model] = {"repetitions": {}}
                results["models"][model]["repetitions"][f"rep_{rep}"] = {
                    "run_id": run_id,
                    "timestamp": datetime.now().isoformat(),
                    "duration_sec": duration_sec,
                    "total_cases": eval_result.total_cases,
                    "passed_cases": eval_result.passed_cases,
                    "failed_cases": eval_result.failed_cases,
                    "accuracy": eval_result.accuracy,
                    "category_accuracy": eval_result.category_accuracy,
                    "priority_accuracy": eval_result.priority_accuracy,
                    "composite_accuracy": eval_result.composite_accuracy,
                    "avg_latency_ms": eval_result.avg_latency_ms,
                    "category_metrics": eval_result.category_metrics,
                    "priority_metrics": eval_result.priority_metrics,
                    "difficulty_breakdown": eval_result.difficulty_breakdown,
                    "tag_breakdown": eval_result.tag_breakdown,
                    "latency_stats": stats["latency"],
                    "reliability_stats": stats["reliability"],
                }
                results["updated_at"] = datetime.now().isoformat()
                save_results(results)

                # Save individual run file
                indiv_run_path = OUTPUT_DIR / f"run_{run_id}_{model_slug}_rep{rep}.json"
                with open(indiv_run_path, "w", encoding="utf-8") as f:
                    json.dump(results["models"][model]["repetitions"][f"rep_{rep}"], f, indent=2)

                print(f"  ✅ [Repetition {rep}/3] Completed in {duration_sec}s | Persisted Run ID: #{run_id}")
                print(f"     Accuracy: {eval_result.accuracy:.2f}% | Cat Acc: {eval_result.category_accuracy:.2f}% | Pri Acc: {eval_result.priority_accuracy:.2f}%")
                print(f"     Avg Latency: {eval_result.avg_latency_ms} ms (Median P50: {stats['latency']['median_ms']} ms)")
                print(f"     Artifact: {indiv_run_path}")

                # 10s cooldown between repetitions
                if rep < 3 and specific_rep is None:
                    print("  ⏳ Cooldown 10s before next repetition...")
                    time.sleep(10)

            except Exception as e:
                print(f"  ❌ [Repetition {rep}/3] FAILED or INTERRUPTED: {e}")
                manifest["runs"][run_key] = {
                    "model": model,
                    "repetition": rep,
                    "status": "failed",
                    "error": str(e),
                    "failed_at": datetime.now().isoformat(),
                }
                save_manifest(manifest)
                raise e

    print("\n" + "=" * 80)
    print("🏁 BENCHMARK EXECUTION CYCLE FINISHED")
    print(f"Manifest written to: {MANIFEST_PATH}")
    print(f"Results written to : {RESULTS_PATH}")
    print("=" * 80 + "\n")


def display_status():
    """Print the current benchmark execution status table from manifest."""
    if not MANIFEST_PATH.exists():
        print("No benchmark manifest found.")
        return

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    print("\n" + "=" * 85)
    print("📊 PROMPTOPS PHASE 6.2 BENCHMARK STATUS")
    print("=" * 85)
    print(f"{'Model':<20} {'Rep':<6} {'Status':<14} {'Run ID':<10} {'Accuracy':<12} {'Avg Latency':<14}")
    print("-" * 85)

    for model in BENCHMARK_MODELS:
        model_slug = model.replace(":", "_").replace("/", "_")
        for rep in [1, 2, 3]:
            key = f"{model_slug}_rep_{rep}"
            info = manifest.get("runs", {}).get(key, {})
            status = info.get("status", "pending")
            run_id = f"#{info.get('run_id')}" if info.get("run_id") else "-"
            acc = f"{info.get('accuracy', 0.0):.2f}%" if "accuracy" in info else "-"
            lat = f"{info.get('avg_latency_ms', 0.0):.1f} ms" if "avg_latency_ms" in info else "-"

            status_str = "✅ Completed" if status == "completed" else ("🔄 In Progress" if status == "in_progress" else ("❌ Failed" if status == "failed" else "⏳ Pending"))
            print(f"{model:<20} {rep:<6} {status_str:<14} {run_id:<10} {acc:<12} {lat:<14}")
    print("=" * 85 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="PromptOps Phase 6.2 Reproducible Multi-Model Benchmark Runner")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1-case pre-flight smoke test on all models")
    parser.add_argument("--status", action="store_true", help="Display current benchmark progress status")
    parser.add_argument("--model", type=str, choices=BENCHMARK_MODELS, help="Run only a specific model")
    parser.add_argument("--repetition", type=int, choices=[1, 2, 3], help="Run only a specific repetition")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Path to golden dataset")
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT, help="Path to prompt configuration")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB, help="Target SQLite database path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.status:
        display_status()
        sys.exit(0)

    if args.smoke_test:
        success = asyncio.run(
            run_smoke_test(
                models=BENCHMARK_MODELS,
                dataset_path=args.dataset,
                prompt_path=args.prompt,
                db_path=args.db_path,
            )
        )
        sys.exit(0 if success else 1)

    target_models = [args.model] if args.model else BENCHMARK_MODELS
    asyncio.run(
        run_benchmark(
            models=target_models,
            specific_rep=args.repetition,
            dataset_path=args.dataset,
            prompt_path=args.prompt,
            db_path=args.db_path,
        )
    )
