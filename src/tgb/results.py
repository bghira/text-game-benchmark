"""JSON result serialization for benchmark runs."""

from __future__ import annotations

import json
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tgb.checks.base import CheckResult
from tgb.clients.ollama_client import TimingData
from tgb.rubric import RubricScore


@dataclass
class ActionResult:
    """Result of a single turn/action within a scenario."""

    action_id: str
    action: str
    checks: list[CheckResult] = field(default_factory=list)
    rubric_scores: list[RubricScore] = field(default_factory=list)
    timing: TimingData | None = None
    raw_response: str = ""

    @property
    def passed(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def failed(self) -> int:
        return sum(1 for c in self.checks if not c.passed)

    @property
    def total(self) -> int:
        return len(self.checks)

    def to_dict(self) -> dict[str, Any]:
        timing_dict = {}
        if self.timing:
            timing_dict = {
                "prompt_tokens": self.timing.prompt_tokens,
                "completion_tokens": self.timing.completion_tokens,
                "eval_tokens_per_sec": self.timing.eval_tokens_per_sec,
                "wall_clock_seconds": self.timing.wall_clock_seconds,
            }
        d: dict[str, Any] = {
            "action_id": self.action_id,
            "action": self.action,
            "checks": [
                {
                    "check_id": c.check_id,
                    "passed": c.passed,
                    "detail": c.detail,
                    "category": c.category,
                }
                for c in self.checks
            ],
            "timing": timing_dict,
            "passed": self.passed,
            "failed": self.failed,
            "total": self.total,
        }
        if self.rubric_scores:
            d["rubric_scores"] = [rs.to_dict() for rs in self.rubric_scores]
        return d


@dataclass
class ScenarioResult:
    """Aggregated result for a scenario run against a specific model."""

    scenario: str
    model: str
    provider: str
    actions: list[ActionResult] = field(default_factory=list)
    rubric_scores: list[RubricScore] = field(default_factory=list)  # scenario-scope scores

    @property
    def total_passed(self) -> int:
        return sum(a.passed for a in self.actions)

    @property
    def total_failed(self) -> int:
        return sum(a.failed for a in self.actions)

    @property
    def total_checks(self) -> int:
        return sum(a.total for a in self.actions)

    @property
    def pass_rate(self) -> float:
        total = self.total_checks
        return self.total_passed / total if total else 0.0

    def by_category(self) -> dict[str, dict[str, int]]:
        """Aggregate results by check category."""
        cats: dict[str, dict[str, int]] = defaultdict(lambda: {"passed": 0, "failed": 0, "total": 0})
        for action in self.actions:
            for check in action.checks:
                cats[check.category]["total"] += 1
                if check.passed:
                    cats[check.category]["passed"] += 1
                else:
                    cats[check.category]["failed"] += 1
        return dict(cats)

    def all_rubric_scores(self) -> list[RubricScore]:
        """All rubric scores: per-turn + scenario-scope."""
        scores = []
        for action in self.actions:
            scores.extend(action.rubric_scores)
        scores.extend(self.rubric_scores)
        return scores

    def rubric_summary(self) -> dict[str, Any]:
        """Aggregate rubric scores by category and rubric."""
        all_scores = self.all_rubric_scores()
        if not all_scores:
            return {}
        by_rubric: dict[str, dict[str, Any]] = {}
        for rs in all_scores:
            if rs.rubric_id not in by_rubric:
                by_rubric[rs.rubric_id] = {
                    "name": rs.rubric_name,
                    "category": rs.category,
                    "scope": rs.scope,
                    "scores": [],
                    "max_score": rs.max_score,
                }
            by_rubric[rs.rubric_id]["scores"].append(rs.score)
            if rs.metric_name:
                by_rubric[rs.rubric_id]["metric_name"] = rs.metric_name
                by_rubric[rs.rubric_id]["metric_value"] = rs.metric_value

        # Compute averages
        summary: dict[str, Any] = {}
        for rid, info in by_rubric.items():
            scores_list = info["scores"]
            valid = [s for s in scores_list if s > 0]
            avg = round(sum(valid) / len(valid), 2) if valid else 0.0
            entry: dict[str, Any] = {
                "name": info["name"],
                "category": info["category"],
                "scope": info["scope"],
                "mean_score": avg,
                "max_score": info["max_score"],
                "n_graded": len(valid),
            }
            if "metric_name" in info:
                entry["metric_name"] = info["metric_name"]
                entry["metric_value"] = info["metric_value"]
            summary[rid] = entry
        return summary

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "scenario": self.scenario,
            "model": self.model,
            "provider": self.provider,
            "actions": [a.to_dict() for a in self.actions],
            "summary": {
                "pass_rate": round(self.pass_rate, 4),
                "passed": self.total_passed,
                "failed": self.total_failed,
                "total": self.total_checks,
                "by_category": self.by_category(),
            },
        }
        rubric_sum = self.rubric_summary()
        if rubric_sum:
            d["summary"]["rubrics"] = rubric_sum
        if self.rubric_scores:
            d["rubric_scores"] = [rs.to_dict() for rs in self.rubric_scores]
        return d


@dataclass
class BenchmarkRun:
    """Complete benchmark run result."""

    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: str = ""
    completed_at: str = ""
    judge_model: str = ""
    results: list[ScenarioResult] = field(default_factory=list)

    def start(self) -> None:
        self.started_at = datetime.now(timezone.utc).isoformat()

    def finish(self) -> None:
        self.completed_at = datetime.now(timezone.utc).isoformat()

    def build_comparison(self) -> dict[str, Any]:
        """Build cross-model comparison table."""
        by_check: dict[str, dict[str, bool | None]] = defaultdict(dict)
        for sr in self.results:
            model_key = sr.model
            for action in sr.actions:
                for check in action.checks:
                    key = f"{sr.scenario}/{action.action_id}/{check.check_id}"
                    by_check[key][model_key] = check.passed
        return {"by_check": dict(by_check)}

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "judge_model": self.judge_model,
            "results": [r.to_dict() for r in self.results],
            "comparison": self.build_comparison(),
        }

    def save(self, output_dir: str | Path) -> Path:
        """Save results to JSON file."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_{timestamp}_{self.run_id[:8]}.json"
        filepath = output_dir / filename

        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

        return filepath


def print_summary(run: BenchmarkRun, verbose: bool = False) -> None:
    """Print a human-readable summary of benchmark results."""
    print(f"\n{'='*60}")
    print(f"Benchmark Run: {run.run_id[:8]}")
    if run.judge_model:
        print(f"Judge Model: {run.judge_model}")
    print(f"{'='*60}")

    for sr in run.results:
        print(f"\n--- {sr.scenario} ({sr.provider}:{sr.model}) ---")
        print(f"Pass rate: {sr.pass_rate:.0%} ({sr.total_passed}/{sr.total_checks})")

        cats = sr.by_category()
        for cat, counts in sorted(cats.items()):
            status = "PASS" if counts["failed"] == 0 else "FAIL"
            print(f"  [{status}] {cat}: {counts['passed']}/{counts['total']}")

        # Rubric summary
        rubric_sum = sr.rubric_summary()
        if rubric_sum:
            print(f"\n  Rubric Grades:")
            for rid, info in sorted(rubric_sum.items()):
                metric_str = ""
                if info.get("metric_name"):
                    metric_str = f" (sim={info['metric_value']:.3f})"
                print(f"    {info['name']}: {info['mean_score']}/{info['max_score']}"
                      f" [{info['category']}]{metric_str}")

        if verbose:
            for action in sr.actions:
                print(f"\n  Action: {action.action_id} ({action.action})")
                if action.timing:
                    print(f"    Timing: {action.timing.wall_clock_seconds}s, "
                          f"{action.timing.eval_tokens_per_sec} tok/s")
                for check in action.checks:
                    icon = "PASS" if check.passed else "FAIL"
                    print(f"    [{icon}] {check.check_id}: {check.detail}")
                for rs in action.rubric_scores:
                    metric_str = ""
                    if rs.metric_name:
                        metric_str = f" (sim={rs.metric_value:.3f})"
                    print(f"    [RUBRIC] {rs.rubric_id}: {rs.score}/{rs.max_score}"
                          f" - {rs.reason}{metric_str}")

            # Scenario-scope rubric scores
            if sr.rubric_scores:
                print(f"\n  Scenario Rubric Grades:")
                for rs in sr.rubric_scores:
                    metric_str = ""
                    if rs.metric_name:
                        metric_str = f" (sim={rs.metric_value:.3f})"
                    print(f"    [RUBRIC] {rs.rubric_id}: {rs.score}/{rs.max_score}"
                          f" - {rs.reason}{metric_str}")

    print(f"\n{'='*60}")
