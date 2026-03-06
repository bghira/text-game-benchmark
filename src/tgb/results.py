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


@dataclass
class ActionResult:
    """Result of a single turn/action within a scenario."""

    action_id: str
    action: str
    checks: list[CheckResult] = field(default_factory=list)
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
        return {
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


@dataclass
class ScenarioResult:
    """Aggregated result for a scenario run against a specific model."""

    scenario: str
    model: str
    provider: str
    actions: list[ActionResult] = field(default_factory=list)

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

    def to_dict(self) -> dict[str, Any]:
        return {
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

        if verbose:
            for action in sr.actions:
                print(f"\n  Action: {action.action_id} ({action.action})")
                if action.timing:
                    print(f"    Timing: {action.timing.wall_clock_seconds}s, "
                          f"{action.timing.eval_tokens_per_sec} tok/s")
                for check in action.checks:
                    icon = "PASS" if check.passed else "FAIL"
                    print(f"    [{icon}] {check.check_id}: {check.detail}")

    print(f"\n{'='*60}")
