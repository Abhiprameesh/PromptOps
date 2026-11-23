import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class RegressionThresholds:
    """
    Configurable thresholds for detecting performance regressions.

    Measurement Semantics & Units:
    - Accuracy drops (accuracy, category_accuracy, priority_accuracy, composite_accuracy):
      Expressed as absolute percentage-point differences on the [0.0, 100.0] scale.
      Example: A drop from 90.0% to 85.0% is a 5.0 percentage-point drop.
    - Overall F1 drops (macro_f1, weighted_f1):
      Expressed as absolute decimal differences on the [0.0, 1.0] scale.
      Example: A drop from 0.90 to 0.80 is a 0.10 drop.
    - Class-level F1 drops (class_f1):
      Expressed as absolute decimal differences on the [0.0, 1.0] scale.
      Example: A per-class drop from 0.85 to 0.75 is a 0.10 drop.
    - Latency increase (avg_latency_ms):
      Expressed as relative percentage increase (%) relative to baseline.
      Example: An increase from 100ms to 150ms is a 50.0% relative increase.
    - Case churn (newly_failed_cases):
      Expressed as absolute count of common test cases that passed in baseline but failed in current.
    """

    accuracy_drop_warning: float = 0.01
    accuracy_drop_critical: float = 5.0
    f1_drop_warning: float = 0.05
    f1_drop_critical: float = 0.10
    class_f1_drop_warning: float = 0.05
    class_f1_drop_critical: float = 0.10
    latency_increase_warning_pct: float = 20.0
    latency_increase_critical_pct: float = 50.0
    newly_failed_cases_warning: int = 1
    newly_failed_cases_critical: int = 3

    def __post_init__(self):
        self.validate()

    def validate(self):
        """Validates threshold values, types, and hierarchy ordering."""
        # Non-negativity and numeric type checks
        numeric_fields = {
            "accuracy_drop_warning": self.accuracy_drop_warning,
            "accuracy_drop_critical": self.accuracy_drop_critical,
            "f1_drop_warning": self.f1_drop_warning,
            "f1_drop_critical": self.f1_drop_critical,
            "class_f1_drop_warning": self.class_f1_drop_warning,
            "class_f1_drop_critical": self.class_f1_drop_critical,
            "latency_increase_warning_pct": self.latency_increase_warning_pct,
            "latency_increase_critical_pct": self.latency_increase_critical_pct,
        }

        for name, val in numeric_fields.items():
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise ValueError(
                    f"Invalid threshold '{name}': must be a numeric value, got {type(val).__name__}"
                )
            if val < 0:
                raise ValueError(
                    f"Invalid threshold '{name}': must be non-negative, got {val}"
                )

        int_fields = {
            "newly_failed_cases_warning": self.newly_failed_cases_warning,
            "newly_failed_cases_critical": self.newly_failed_cases_critical,
        }
        for name, val in int_fields.items():
            if not isinstance(val, int) or isinstance(val, bool):
                raise ValueError(
                    f"Invalid threshold '{name}': must be an integer, got {type(val).__name__}"
                )
            if val < 0:
                raise ValueError(
                    f"Invalid threshold '{name}': must be non-negative, got {val}"
                )

        # Hierarchy validation: Warning must not exceed Critical
        if self.accuracy_drop_warning > self.accuracy_drop_critical:
            raise ValueError(
                f"Invalid threshold hierarchy: accuracy_drop_warning ({self.accuracy_drop_warning}) "
                f"cannot exceed accuracy_drop_critical ({self.accuracy_drop_critical})"
            )
        if self.f1_drop_warning > self.f1_drop_critical:
            raise ValueError(
                f"Invalid threshold hierarchy: f1_drop_warning ({self.f1_drop_warning}) "
                f"cannot exceed f1_drop_critical ({self.f1_drop_critical})"
            )
        if self.class_f1_drop_warning > self.class_f1_drop_critical:
            raise ValueError(
                f"Invalid threshold hierarchy: class_f1_drop_warning ({self.class_f1_drop_warning}) "
                f"cannot exceed class_f1_drop_critical ({self.class_f1_drop_critical})"
            )
        if self.latency_increase_warning_pct > self.latency_increase_critical_pct:
            raise ValueError(
                f"Invalid threshold hierarchy: latency_increase_warning_pct ({self.latency_increase_warning_pct}) "
                f"cannot exceed latency_increase_critical_pct ({self.latency_increase_critical_pct})"
            )
        if self.newly_failed_cases_warning > self.newly_failed_cases_critical:
            raise ValueError(
                f"Invalid threshold hierarchy: newly_failed_cases_warning ({self.newly_failed_cases_warning}) "
                f"cannot exceed newly_failed_cases_critical ({self.newly_failed_cases_critical})"
            )

    @classmethod
    def from_env(cls) -> "RegressionThresholds":
        """
        Load thresholds from environment variables with strict numeric validation.
        Produces actionable errors on malformed environment values instead of silent fallback.
        """
        def _parse_float(name: str, default: float) -> float:
            raw = os.getenv(name)
            if raw is None or raw.strip() == "":
                return default
            try:
                return float(raw.strip())
            except ValueError:
                raise ValueError(
                    f"Invalid environment configuration: '{name}' must be a valid float, got '{raw}'"
                )

        def _parse_int(name: str, default: int) -> int:
            raw = os.getenv(name)
            if raw is None or raw.strip() == "":
                return default
            try:
                val = float(raw.strip())
                if not val.is_integer():
                    raise ValueError()
                return int(val)
            except ValueError:
                raise ValueError(
                    f"Invalid environment configuration: '{name}' must be a valid integer, got '{raw}'"
                )

        thresholds = cls(
            accuracy_drop_warning=_parse_float("PROMPTOPS_ACCURACY_DROP_WARNING", 0.01),
            accuracy_drop_critical=_parse_float("PROMPTOPS_ACCURACY_DROP_CRITICAL", 5.0),
            f1_drop_warning=_parse_float("PROMPTOPS_F1_DROP_WARNING", 0.05),
            f1_drop_critical=_parse_float("PROMPTOPS_F1_DROP_CRITICAL", 0.10),
            class_f1_drop_warning=_parse_float("PROMPTOPS_CLASS_F1_DROP_WARNING", 0.05),
            class_f1_drop_critical=_parse_float("PROMPTOPS_CLASS_F1_DROP_CRITICAL", 0.10),
            latency_increase_warning_pct=_parse_float("PROMPTOPS_LATENCY_WARNING_PCT", 20.0),
            latency_increase_critical_pct=_parse_float("PROMPTOPS_LATENCY_CRITICAL_PCT", 50.0),
            newly_failed_cases_warning=_parse_int("PROMPTOPS_NEWLY_FAILED_WARNING", 1),
            newly_failed_cases_critical=_parse_int("PROMPTOPS_NEWLY_FAILED_CRITICAL", 3),
        )
        thresholds.validate()
        return thresholds


def _safe_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


class RegressionDetector:

    @classmethod
    def compare_from_db(
        cls,
        db: Any,
        previous_run_id: Optional[int] = None,
        current_run_id: Optional[int] = None,
        thresholds: Optional[RegressionThresholds] = None,
    ) -> Dict[str, Any]:
        """
        Loads runs and cases from the database and performs comprehensive V2 regression analysis.
        If run IDs are not provided, defaults to the two most recent runs.
        """
        if current_run_id is None or previous_run_id is None:
            last_two = db.get_last_two_runs()
            if len(last_two) < 2:
                raise ValueError("Need at least two evaluation runs for comparison.")
            curr_id = last_two[0][0]
            prev_id = last_two[1][0]
            current_run_id = current_run_id or curr_id
            previous_run_id = previous_run_id or prev_id

        # Attempt to load extended run tuples (falls back to legacy if unavailable)
        prev_run = (
            db.get_extended_run_by_id(previous_run_id)
            if hasattr(db, "get_extended_run_by_id")
            else None
        )
        curr_run = (
            db.get_extended_run_by_id(current_run_id)
            if hasattr(db, "get_extended_run_by_id")
            else None
        )

        if prev_run is None:
            prev_run = db.get_run_by_id(previous_run_id)
        if curr_run is None:
            curr_run = db.get_run_by_id(current_run_id)

        # Load case results
        prev_cases = (
            db.get_extended_case_results(previous_run_id)
            if hasattr(db, "get_extended_case_results")
            else db.get_case_results(previous_run_id)
        )
        curr_cases = (
            db.get_extended_case_results(current_run_id)
            if hasattr(db, "get_extended_case_results")
            else db.get_case_results(current_run_id)
        )

        return cls.compare(
            previous_run=prev_run,
            current_run=curr_run,
            previous_cases=prev_cases,
            current_cases=curr_cases,
            thresholds=thresholds,
        )

    @classmethod
    def compare(
        cls,
        previous_run: Any,
        current_run: Any,
        previous_cases: Optional[List[Any]] = None,
        current_cases: Optional[List[Any]] = None,
        thresholds: Optional[RegressionThresholds] = None,
    ) -> Dict[str, Any]:
        """
        Compares two runs across multi-field accuracies, F1 metrics, latency,
        case-level churn, and class-level changes against configurable thresholds.
        """
        thresholds = thresholds or RegressionThresholds()

        # Extract run metadata and basic accuracies
        prev_run_id = previous_run[0]
        curr_run_id = current_run[0]
        prev_ts = previous_run[1]
        curr_ts = current_run[1]
        prompt_version = current_run[2]
        model = current_run[3]

        # Extract accuracy (handles 5-element, 8-element, and 16-element tuples)
        prev_acc = (
            previous_run[4]
            if len(previous_run) == 5
            else previous_run[7]
        )
        curr_acc = (
            current_run[4]
            if len(current_run) == 5
            else current_run[7]
        )
        accuracy_diff = round(curr_acc - prev_acc, 2)

        # Extended metrics extraction (handles missing historical metrics safely)
        metrics_comp: Dict[str, Dict[str, Any]] = {}
        degraded_metrics: List[Dict[str, Any]] = []
        improved_metrics: List[Dict[str, Any]] = []

        def _evaluate_metric(
            name: str,
            prev_val: Optional[float],
            curr_val: Optional[float],
            higher_is_better: bool = True,
            warning_delta: float = 0.01,
            critical_delta: float = 5.0,
            is_percentage: bool = False,
        ):
            if prev_val is None or curr_val is None:
                metrics_comp[name] = {
                    "previous": prev_val,
                    "current": curr_val,
                    "delta": None,
                    "status": "unavailable",
                }
                return

            delta = round(curr_val - prev_val, 4)
            pct_change = (
                round(((curr_val - prev_val) / prev_val) * 100.0, 2)
                if prev_val > 0
                else 0.0
            )

            is_degraded = False
            is_critical = False
            is_warning = False
            is_improved = False

            if higher_is_better:
                if delta < 0:
                    is_degraded = True
                    drop = abs(delta)
                    if drop >= critical_delta:
                        is_critical = True
                    elif drop >= warning_delta:
                        is_warning = True
                elif delta > 0:
                    is_improved = True
            else:
                # Lower is better (e.g. latency)
                if delta > 0:
                    is_degraded = True
                    increase = pct_change if is_percentage else delta
                    if increase >= critical_delta:
                        is_critical = True
                    elif increase >= warning_delta:
                        is_warning = True
                elif delta < 0:
                    is_improved = True

            status = (
                "critical"
                if is_critical
                else "warning"
                if is_warning
                else "improved"
                if is_improved
                else "unchanged"
            )

            metric_entry = {
                "previous": prev_val,
                "current": curr_val,
                "delta": delta,
                "status": status,
            }
            if is_percentage:
                metric_entry["pct_change"] = pct_change

            metrics_comp[name] = metric_entry

            if is_critical or is_warning:
                degraded_metrics.append({
                    "metric": name,
                    "previous": prev_val,
                    "current": curr_val,
                    "delta": delta,
                    "severity": "critical" if is_critical else "warning",
                })
            elif is_improved:
                improved_metrics.append({
                    "metric": name,
                    "previous": prev_val,
                    "current": curr_val,
                    "delta": delta,
                })

        # Overall Accuracy (percentage points)
        _evaluate_metric(
            "accuracy",
            prev_acc,
            curr_acc,
            higher_is_better=True,
            warning_delta=thresholds.accuracy_drop_warning,
            critical_delta=thresholds.accuracy_drop_critical,
        )

        # Extended fields if available (length >= 12)
        has_extended = len(previous_run) >= 12 and len(current_run) >= 12

        prev_cat_acc = _safe_float(previous_run[8]) if has_extended else None
        curr_cat_acc = _safe_float(current_run[8]) if has_extended else None
        _evaluate_metric(
            "category_accuracy",
            prev_cat_acc,
            curr_cat_acc,
            higher_is_better=True,
            warning_delta=thresholds.accuracy_drop_warning,
            critical_delta=thresholds.accuracy_drop_critical,
        )

        prev_pri_acc = _safe_float(previous_run[9]) if has_extended else None
        curr_pri_acc = _safe_float(current_run[9]) if has_extended else None
        _evaluate_metric(
            "priority_accuracy",
            prev_pri_acc,
            curr_pri_acc,
            higher_is_better=True,
            warning_delta=thresholds.accuracy_drop_warning,
            critical_delta=thresholds.accuracy_drop_critical,
        )

        prev_comp_acc = _safe_float(previous_run[10]) if has_extended else None
        curr_comp_acc = _safe_float(current_run[10]) if has_extended else None
        _evaluate_metric(
            "composite_accuracy",
            prev_comp_acc,
            curr_comp_acc,
            higher_is_better=True,
            warning_delta=thresholds.accuracy_drop_warning,
            critical_delta=thresholds.accuracy_drop_critical,
        )

        prev_lat = _safe_float(previous_run[11]) if has_extended else None
        curr_lat = _safe_float(current_run[11]) if has_extended else None
        _evaluate_metric(
            "avg_latency_ms",
            prev_lat,
            curr_lat,
            higher_is_better=False,
            warning_delta=thresholds.latency_increase_warning_pct,
            critical_delta=thresholds.latency_increase_critical_pct,
            is_percentage=True,
        )

        # F1 Metrics extraction (decimal differences on [0.0, 1.0])
        prev_cat_m = previous_run[12] if len(previous_run) > 12 else None
        curr_cat_m = current_run[12] if len(current_run) > 12 else None
        prev_pri_m = previous_run[13] if len(previous_run) > 13 else None
        curr_pri_m = current_run[13] if len(current_run) > 13 else None

        prev_cat_macro = (
            _safe_float(prev_cat_m.get("macro_avg", {}).get("f1"))
            if isinstance(prev_cat_m, dict)
            else None
        )
        curr_cat_macro = (
            _safe_float(curr_cat_m.get("macro_avg", {}).get("f1"))
            if isinstance(curr_cat_m, dict)
            else None
        )
        _evaluate_metric(
            "category_macro_f1",
            prev_cat_macro,
            curr_cat_macro,
            higher_is_better=True,
            warning_delta=thresholds.f1_drop_warning,
            critical_delta=thresholds.f1_drop_critical,
        )

        prev_cat_weighted = (
            _safe_float(prev_cat_m.get("weighted_avg", {}).get("f1"))
            if isinstance(prev_cat_m, dict)
            else None
        )
        curr_cat_weighted = (
            _safe_float(curr_cat_m.get("weighted_avg", {}).get("f1"))
            if isinstance(curr_cat_m, dict)
            else None
        )
        _evaluate_metric(
            "category_weighted_f1",
            prev_cat_weighted,
            curr_cat_weighted,
            higher_is_better=True,
            warning_delta=thresholds.f1_drop_warning,
            critical_delta=thresholds.f1_drop_critical,
        )

        prev_pri_macro = (
            _safe_float(prev_pri_m.get("macro_avg", {}).get("f1"))
            if isinstance(prev_pri_m, dict)
            else None
        )
        curr_pri_macro = (
            _safe_float(curr_pri_m.get("macro_avg", {}).get("f1"))
            if isinstance(curr_pri_m, dict)
            else None
        )
        _evaluate_metric(
            "priority_macro_f1",
            prev_pri_macro,
            curr_pri_macro,
            higher_is_better=True,
            warning_delta=thresholds.f1_drop_warning,
            critical_delta=thresholds.f1_drop_critical,
        )

        # ---------------------------------------------------------
        # Case-Level Churn Analysis with Edge Case Handling
        # ---------------------------------------------------------
        case_analysis = {
            "newly_failed": [],
            "recovered": [],
            "unchanged_passing": [],
            "unchanged_failing": [],
            "added_cases": [],
            "removed_cases": [],
            "missing_pass_cases": [],
            "duplicate_cases": [],
        }

        if previous_cases is not None and current_cases is not None:
            # 1. Detect duplicate case IDs explicitly
            prev_counts: Dict[str, int] = {}
            for row in previous_cases:
                cid = row[0]
                prev_counts[cid] = prev_counts.get(cid, 0) + 1

            curr_counts: Dict[str, int] = {}
            for row in current_cases:
                cid = row[0]
                curr_counts[cid] = curr_counts.get(cid, 0) + 1

            duplicate_ids = []
            for cid, count in prev_counts.items():
                if count > 1 and cid not in duplicate_ids:
                    duplicate_ids.append(cid)
            for cid, count in curr_counts.items():
                if count > 1 and cid not in duplicate_ids:
                    duplicate_ids.append(cid)
            case_analysis["duplicate_cases"] = duplicate_ids

            # 2. Build unique lookup maps (first occurrence wins)
            prev_map: Dict[str, Any] = {}
            for row in previous_cases:
                cid = row[0]
                if cid not in prev_map:
                    prev_map[cid] = row[3]

            curr_map: Dict[str, Any] = {}
            for row in current_cases:
                cid = row[0]
                if cid not in curr_map:
                    curr_map[cid] = row[3]

            # 3. Identify Added and Removed cases (reported separately from churn)
            for cid in curr_map:
                if cid not in prev_map:
                    case_analysis["added_cases"].append(cid)

            for cid in prev_map:
                if cid not in curr_map:
                    case_analysis["removed_cases"].append(cid)

            # 4. Classify common test cases
            for cid, curr_pass in curr_map.items():
                if cid in prev_map:
                    prev_pass = prev_map[cid]

                    # Missing pass/fail information check
                    if prev_pass is None or curr_pass is None:
                        case_analysis["missing_pass_cases"].append(cid)
                        continue

                    prev_bool = bool(prev_pass)
                    curr_bool = bool(curr_pass)

                    if prev_bool and not curr_bool:
                        case_analysis["newly_failed"].append(cid)
                    elif not prev_bool and curr_bool:
                        case_analysis["recovered"].append(cid)
                    elif prev_bool and curr_bool:
                        case_analysis["unchanged_passing"].append(cid)
                    else:
                        case_analysis["unchanged_failing"].append(cid)

        newly_failed_count = len(case_analysis["newly_failed"])
        recovered_count = len(case_analysis["recovered"])
        affected_cases = case_analysis["newly_failed"]

        # Check newly failed cases against thresholds
        if newly_failed_count >= thresholds.newly_failed_cases_critical:
            degraded_metrics.append({
                "metric": "newly_failed_cases",
                "count": newly_failed_count,
                "severity": "critical",
            })
        elif newly_failed_count >= thresholds.newly_failed_cases_warning:
            degraded_metrics.append({
                "metric": "newly_failed_cases",
                "count": newly_failed_count,
                "severity": "warning",
            })

        # ---------------------------------------------------------
        # Class-Level F1 Comparison (Integrated into Severity)
        # ---------------------------------------------------------
        class_analysis = {"category": {}, "priority": {}}

        def _compare_classes(target: str, prev_m: Any, curr_m: Any):
            if isinstance(prev_m, dict) and isinstance(curr_m, dict):
                p_classes = prev_m.get("per_class", {})
                c_classes = curr_m.get("per_class", {})
                all_classes = sorted(list(set(p_classes.keys()).union(set(c_classes.keys()))))

                for cls_name in all_classes:
                    f1_p = _safe_float(p_classes.get(cls_name, {}).get("f1"))
                    f1_c = _safe_float(c_classes.get(cls_name, {}).get("f1"))

                    if f1_p is None or f1_c is None:
                        class_analysis[target][cls_name] = {
                            "previous_f1": f1_p,
                            "current_f1": f1_c,
                            "delta": None,
                            "status": "unavailable",
                        }
                        continue

                    diff = round(f1_c - f1_p, 4)
                    drop = abs(diff)

                    if diff < 0:
                        if drop >= thresholds.class_f1_drop_critical:
                            cls_status = "critical"
                            degraded_metrics.append({
                                "metric": f"{target}_class_{cls_name}_f1",
                                "target": target,
                                "class": cls_name,
                                "previous": f1_p,
                                "current": f1_c,
                                "delta": diff,
                                "severity": "critical",
                            })
                        elif drop >= thresholds.class_f1_drop_warning:
                            cls_status = "warning"
                            degraded_metrics.append({
                                "metric": f"{target}_class_{cls_name}_f1",
                                "target": target,
                                "class": cls_name,
                                "previous": f1_p,
                                "current": f1_c,
                                "delta": diff,
                                "severity": "warning",
                            })
                        else:
                            cls_status = "unchanged"
                    elif diff >= thresholds.class_f1_drop_warning:
                        cls_status = "improved"
                        improved_metrics.append({
                            "metric": f"{target}_class_{cls_name}_f1",
                            "target": target,
                            "class": cls_name,
                            "previous": f1_p,
                            "current": f1_c,
                            "delta": diff,
                        })
                    else:
                        cls_status = "unchanged"

                    class_analysis[target][cls_name] = {
                        "previous_f1": f1_p,
                        "current_f1": f1_c,
                        "delta": diff,
                        "status": cls_status,
                    }

        _compare_classes("category", prev_cat_m, curr_cat_m)
        _compare_classes("priority", prev_pri_m, curr_pri_m)

        # ---------------------------------------------------------
        # Final Severity & Action Aggregation
        # ---------------------------------------------------------
        has_critical = any(
            d.get("severity") == "critical" for d in degraded_metrics
        )
        has_warning = any(
            d.get("severity") == "warning" for d in degraded_metrics
        )

        if has_critical:
            overall_severity = "critical"
            status = "regression"
            action = (
                f"BLOCKED: Critical regression detected. "
                f"{len(affected_cases)} case(s) broke and "
                f"{len(degraded_metrics)} metric(s)/class(es) severely degraded."
            )
        elif has_warning:
            overall_severity = "warning"
            status = "warning"
            action = (
                f"WARNING: Performance degradation detected within warning limits. "
                f"{newly_failed_count} newly failed case(s), {len(degraded_metrics)} warning metric(s). "
                f"Review before deployment."
            )
        elif improved_metrics and not degraded_metrics:
            overall_severity = "none"
            status = "improved"
            action = (
                f"APPROVED: Performance improved (+{accuracy_diff}% accuracy, "
                f"{recovered_count} case(s) recovered). Safe to deploy."
            )
        else:
            overall_severity = "none"
            status = "unchanged"
            action = "UNCHANGED: Performance metrics are consistent with baseline."

        return {
            # Legacy compatible keys
            "status": status,
            "difference": accuracy_diff,
            "previous_accuracy": prev_acc,
            "current_accuracy": curr_acc,
            "previous_run": prev_run_id,
            "current_run": curr_run_id,
            "previous_timestamp": prev_ts,
            "current_timestamp": curr_ts,
            "prompt_version": prompt_version,
            "model": model,
            # Evaluation Engine V2 Analytics
            "severity": overall_severity,
            "metrics_comparison": metrics_comp,
            "case_analysis": case_analysis,
            "class_analysis": class_analysis,
            "degraded_metrics": degraded_metrics,
            "improved_metrics": improved_metrics,
            "affected_cases": affected_cases,
            "recommended_action": action,
            "thresholds": thresholds.__dict__,
        }
