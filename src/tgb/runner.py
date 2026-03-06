"""Scenario runner — orchestrates scenario → model → judge → results."""

from __future__ import annotations

import sys
from typing import Any, Protocol

from tgb.checks.base import CheckResult
from tgb.checks.registry import get_check
from tgb.config import Scenario, TurnSpec
from tgb.judge import JudgeEvaluator
from tgb.prompt_builder import AccumulatedState, PromptBuilder
from tgb.response_parser import ParsedResponse, parse_response
from tgb.results import ActionResult, ScenarioResult
from tgb.clients.ollama_client import TimingData


class CompletionClient(Protocol):
    """Protocol for model completion clients."""

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        **opts: Any,
    ) -> tuple[str, TimingData]: ...


def run_auto_checks(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
) -> list[CheckResult]:
    """Run all automated (non-judge) checks for a turn."""
    results = []
    for check_spec in turn.checks:
        if check_spec.check_id.startswith("judge:"):
            continue
        try:
            check_fn = get_check(check_spec.check_id)
            result = check_fn(parsed, scenario, turn, state, check_spec.params)
            results.append(result)
        except KeyError as e:
            results.append(CheckResult(
                check_id=check_spec.check_id,
                passed=False,
                detail=f"Unknown check: {e}",
                category="error",
            ))
        except Exception as e:
            results.append(CheckResult(
                check_id=check_spec.check_id,
                passed=False,
                detail=f"Check error: {e}",
                category="error",
            ))
    return results


def run_scenario(
    scenario: Scenario,
    client: CompletionClient,
    provider: str,
    model: str,
    judge: JudgeEvaluator | None = None,
    verbose: bool = False,
) -> ScenarioResult:
    """Run a complete scenario, returning aggregated results."""
    builder = PromptBuilder()
    state = AccumulatedState(scenario)
    scenario_result = ScenarioResult(
        scenario=scenario.name,
        model=model,
        provider=provider,
    )

    for i, turn in enumerate(scenario.turns):
        if verbose:
            print(f"  Turn {i+1}/{len(scenario.turns)}: {turn.action}", file=sys.stderr)

        # Build prompt
        system_prompt, user_prompt = builder.build(scenario, turn, state)

        # Get model response
        try:
            raw_text, timing = client.complete(system_prompt, user_prompt)
        except Exception as e:
            # Model failure — create a failed action result
            action_result = ActionResult(
                action_id=turn.action_id,
                action=turn.action,
                checks=[CheckResult(
                    check_id="model_completion",
                    passed=False,
                    detail=f"Model error: {e}",
                    category="error",
                )],
                timing=TimingData(),
                raw_response="",
            )
            scenario_result.actions.append(action_result)
            continue

        # Parse response
        parsed = parse_response(raw_text)

        if verbose:
            status = "tool_call" if parsed.is_tool_call else "response"
            print(f"    Parsed: {status}, keys={list(parsed.parsed_json.keys())[:5]}", file=sys.stderr)

        # Run automated checks
        auto_results = run_auto_checks(parsed, scenario, turn, state)

        # Run judge checks
        judge_results: list[CheckResult] = []
        if judge:
            judge_results = judge.evaluate(parsed, scenario, turn, state)

        all_results = auto_results + judge_results

        # Create action result
        action_result = ActionResult(
            action_id=turn.action_id,
            action=turn.action,
            checks=all_results,
            timing=timing,
            raw_response=raw_text,
        )
        scenario_result.actions.append(action_result)

        if verbose:
            passed = sum(1 for r in all_results if r.passed)
            total = len(all_results)
            print(f"    Checks: {passed}/{total} passed", file=sys.stderr)

        # Update accumulated state for next turn
        state.apply(parsed.parsed_json if not parsed.is_tool_call else None)

    return scenario_result
