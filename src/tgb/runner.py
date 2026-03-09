"""Scenario runner — orchestrates scenario → model → judge → results.

Supports the engine's multi-round tool-call flow: the model may return
tool_call responses (recent_turns, ready_to_write, memory_search, etc.)
before emitting the final narration JSON. The runner loops up to
MAX_TOOL_ROUNDS times, feeding synthetic tool results back, before
collecting the final response for checks and grading.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Protocol

from tgb.checks.base import CheckResult
from tgb.checks.registry import get_check
from tgb.config import Scenario, TurnSpec
from tgb.judge import JudgeEvaluator
from tgb.prompt_builder import AccumulatedState, PromptBuilder
from tgb.response_parser import ParsedResponse, parse_response
from tgb.results import ActionResult, ScenarioResult
from tgb.rubric import Rubric, RubricGrader
from tgb.clients.ollama_client import TimingData

# Maximum tool-call rounds before forcing final response
MAX_TOOL_ROUNDS = 4


class CompletionClient(Protocol):
    """Protocol for model completion clients."""

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        **opts: Any,
    ) -> tuple[str, TimingData]: ...


# ── Synthetic tool results ─────────────────────────────────────────

def _synthetic_tool_result(parsed_json: dict[str, Any], state: AccumulatedState) -> str:
    """Generate a synthetic tool result for a tool_call response.

    In the real engine, tool calls hit the database/memory layer. For
    benchmarking we return plausible stub results so the model can
    proceed to the final narration round.
    """
    tool = parsed_json.get("tool_call", "")

    if tool == "recent_turns":
        # Return existing recent_turns from scenario state
        turns = []
        for rt in state.recent_turns[-8:]:
            turns.append({
                "tag": rt.get("tag", ""),
                "action": rt.get("action", ""),
                "narration": rt.get("narration", ""),
            })
        return json.dumps({
            "tool_result": "recent_turns",
            "turns": turns,
            "count": len(turns),
        }, ensure_ascii=False)

    if tool == "ready_to_write":
        return json.dumps({
            "tool_result": "ready_to_write",
            "status": "ok",
            "instruction": "Proceed with final narration and state JSON.",
        }, ensure_ascii=False)

    if tool == "memory_search":
        return json.dumps({
            "tool_result": "memory_search",
            "results": [],
            "note": "No memories found (benchmark stub).",
        }, ensure_ascii=False)

    if tool in ("sms_list", "sms_read"):
        return json.dumps({
            "tool_result": tool,
            "threads": [],
            "messages": [],
        }, ensure_ascii=False)

    if tool == "sms_write":
        return json.dumps({
            "tool_result": "sms_write",
            "status": "sent",
        }, ensure_ascii=False)

    if tool == "sms_schedule":
        return json.dumps({
            "tool_result": "sms_schedule",
            "status": "scheduled",
        }, ensure_ascii=False)

    if tool in ("memory_terms", "memory_turn", "memory_store"):
        return json.dumps({
            "tool_result": tool,
            "results": [],
        }, ensure_ascii=False)

    if tool == "source_browse":
        return json.dumps({
            "tool_result": "source_browse",
            "keys": [],
            "note": "No source material loaded (benchmark stub).",
        }, ensure_ascii=False)

    if tool == "name_generate":
        return json.dumps({
            "tool_result": "name_generate",
            "names": [{"first": "Alex", "last": "Morgan", "origin": "english"}],
        }, ensure_ascii=False)

    if tool == "story_outline":
        return json.dumps({
            "tool_result": "story_outline",
            "outline": {},
        }, ensure_ascii=False)

    if tool in ("plot_plan", "chapter_plan", "consequence_log"):
        return json.dumps({
            "tool_result": tool,
            "status": "recorded",
        }, ensure_ascii=False)

    # Unknown tool — generic ack
    return json.dumps({
        "tool_result": tool,
        "status": "ok",
    }, ensure_ascii=False)


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


def _merge_timing(base: TimingData, addition: TimingData) -> TimingData:
    """Merge timing data from multiple rounds (accumulate tokens and time)."""
    return TimingData(
        prompt_tokens=(base.prompt_tokens or 0) + (addition.prompt_tokens or 0),
        completion_tokens=(base.completion_tokens or 0) + (addition.completion_tokens or 0),
        eval_tokens_per_sec=addition.eval_tokens_per_sec or base.eval_tokens_per_sec,
        wall_clock_seconds=(base.wall_clock_seconds or 0) + (addition.wall_clock_seconds or 0),
    )


def run_scenario(
    scenario: Scenario,
    client: CompletionClient,
    provider: str,
    model: str,
    judge: JudgeEvaluator | None = None,
    rubric_grader: RubricGrader | None = None,
    rubrics: list[Rubric] | None = None,
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
    narrations: list[str] = []  # collected for rubric grading

    for i, turn in enumerate(scenario.turns):
        if verbose:
            print(f"  Turn {i+1}/{len(scenario.turns)}: {turn.action}", file=sys.stderr)

        # Build prompt
        system_prompt, user_prompt = builder.build(scenario, turn, state)

        # Get model response — with tool-call loop
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
            narrations.append("")
            continue

        parsed = parse_response(raw_text)
        tool_round = 0
        accumulated_timing = timing

        # Tool-call loop: if the model returns a tool_call, feed back
        # a synthetic result and re-call until we get a final response
        while parsed.is_tool_call and tool_round < MAX_TOOL_ROUNDS:
            tool_name = parsed.parsed_json.get("tool_call", "unknown")

            if verbose:
                print(f"    Tool round {tool_round+1}: {tool_name}", file=sys.stderr)

            # Track tool calls in state (for subplot/SMS analysis)
            state.apply(parsed.parsed_json)

            # Generate synthetic tool result
            tool_result = _synthetic_tool_result(parsed.parsed_json, state)

            # Append tool result to user prompt and re-call
            user_prompt = (
                f"{user_prompt}\n"
                f"ASSISTANT_RESPONSE:\n{raw_text}\n"
                f"TOOL_RESULT:\n{tool_result}\n"
            )

            try:
                raw_text, timing = client.complete(system_prompt, user_prompt)
            except Exception as e:
                # Model failure mid-tool-loop
                parsed = ParsedResponse(
                    raw=str(e),
                    parse_error=f"Model error during tool round {tool_round+1}: {e}",
                )
                break

            accumulated_timing = _merge_timing(accumulated_timing, timing)
            parsed = parse_response(raw_text)
            tool_round += 1

        if verbose:
            status = "tool_call" if parsed.is_tool_call else "response"
            extra = f" (after {tool_round} tool rounds)" if tool_round > 0 else ""
            print(f"    Parsed: {status}{extra}, keys={list(parsed.parsed_json.keys())[:5]}", file=sys.stderr)

        # Collect narration for rubric grading
        narration = parsed.parsed_json.get("narration", "")
        if isinstance(narration, str):
            narrations.append(narration)
        else:
            narrations.append("")

        # Run automated checks
        auto_results = run_auto_checks(parsed, scenario, turn, state)

        # Run judge checks
        judge_results: list[CheckResult] = []
        if judge:
            judge_results = judge.evaluate(parsed, scenario, turn, state)

        all_results = auto_results + judge_results

        # Grade turn-scope rubrics
        rubric_scores = []
        if rubric_grader and rubrics:
            rubric_scores = rubric_grader.grade_turn(
                rubrics, parsed, scenario, turn, state, narrations,
            )

        # Create action result
        action_result = ActionResult(
            action_id=turn.action_id,
            action=turn.action,
            checks=all_results,
            rubric_scores=rubric_scores,
            timing=accumulated_timing,
            raw_response=raw_text,
        )
        scenario_result.actions.append(action_result)

        if verbose:
            passed = sum(1 for r in all_results if r.passed)
            total = len(all_results)
            print(f"    Checks: {passed}/{total} passed", file=sys.stderr)
            if rubric_scores:
                for rs in rubric_scores:
                    print(f"    Rubric {rs.rubric_id}: {rs.score}/{rs.max_score}", file=sys.stderr)

        # Update accumulated state for next turn (final response only)
        state.apply(parsed.parsed_json if not parsed.is_tool_call else None)

    # Grade scenario-scope rubrics (across all turns)
    if rubric_grader and rubrics and narrations:
        scenario_rubric_scores = rubric_grader.grade_scenario(
            rubrics, narrations, scenario, state,
        )
        scenario_result.rubric_scores = scenario_rubric_scores
        if verbose and scenario_rubric_scores:
            for rs in scenario_rubric_scores:
                metric_str = f" (sim={rs.metric_value:.3f})" if rs.metric_name else ""
                print(f"  Scenario rubric {rs.rubric_id}: "
                      f"{rs.score}/{rs.max_score}{metric_str}", file=sys.stderr)

    return scenario_result
