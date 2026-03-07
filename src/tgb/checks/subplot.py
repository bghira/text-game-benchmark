"""Subplot checks — validates plot threads, chapters, and consequences.

The engine has three narrative planning tools:
- plot_plan: multi-turn threads with target_turns (1-250)
- chapter_plan: emergent scene structure with scene lists
- consequence_log: persistent effects with expires_turns

These checks validate structural correctness and self-consistency:
does the model follow through on its own declared plans?
"""

from __future__ import annotations

import re
from typing import Any

from tgb.checks.base import CheckResult
from tgb.config import Scenario, TurnSpec
from tgb.prompt_builder import AccumulatedState
from tgb.response_parser import ParsedResponse


def _slug_valid(slug: str) -> bool:
    """Check if a string is a valid kebab-case slug."""
    return bool(re.match(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$", slug))


# ── Plot thread checks ──────────────────────────────────────────────────


def plot_thread_fields_valid(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Validate plot_plan tool call fields against engine contract."""
    if parsed.parsed_json.get("tool_call") != "plot_plan":
        return CheckResult(
            check_id="plot_thread_fields_valid",
            passed=True,
            detail="Not a plot_plan tool call, skipped",
            category="subplot",
        )

    plans = parsed.parsed_json.get("plans", [])
    if not isinstance(plans, list) or not plans:
        return CheckResult(
            check_id="plot_thread_fields_valid",
            passed=False,
            detail="plot_plan missing or empty 'plans' array",
            category="subplot",
        )

    issues = []
    for i, plan in enumerate(plans):
        if not isinstance(plan, dict):
            issues.append(f"plans[{i}] is not a dict")
            continue

        thread = plan.get("thread", "")
        if not thread:
            issues.append(f"plans[{i}] missing 'thread' slug")
        elif not _slug_valid(thread):
            issues.append(f"plans[{i}] thread slug '{thread}' not valid kebab-case")

        # target_turns validation
        target = plan.get("target_turns")
        if target is not None:
            if not isinstance(target, (int, float)):
                issues.append(f"plans[{i}] target_turns not numeric: {target}")
            elif int(target) < 1 or int(target) > 250:
                issues.append(f"plans[{i}] target_turns {target} out of range [1, 250]")

        # Field length limits
        for field, limit in [("setup", 260), ("intended_payoff", 260), ("resolution", 260)]:
            val = plan.get(field, "")
            if isinstance(val, str) and len(val) > limit:
                issues.append(f"plans[{i}] {field} exceeds {limit} chars ({len(val)})")

        # Status validation
        status = plan.get("status")
        if status is not None and status not in ("active", "resolved"):
            issues.append(f"plans[{i}] status '{status}' not 'active' or 'resolved'")

        # Dependencies validation
        deps = plan.get("dependencies")
        if deps is not None:
            if not isinstance(deps, list):
                issues.append(f"plans[{i}] dependencies not a list")
            elif len(deps) > 8:
                issues.append(f"plans[{i}] dependencies exceeds 8 ({len(deps)})")

    if issues:
        return CheckResult(
            check_id="plot_thread_fields_valid",
            passed=False,
            detail="; ".join(issues),
            category="subplot",
        )
    return CheckResult(
        check_id="plot_thread_fields_valid",
        passed=True,
        detail=f"Valid plot_plan with {len(plans)} thread(s)",
        category="subplot",
    )


def plot_thread_target_reasonable(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that target_turns is reasonable for the thread scope.

    A plot thread with target_turns=1 is suspicious (use a consequence instead).
    A plot thread with target_turns > 100 in a short scenario is suspicious.
    """
    if parsed.parsed_json.get("tool_call") != "plot_plan":
        return CheckResult(
            check_id="plot_thread_target_reasonable",
            passed=True,
            detail="Not a plot_plan tool call, skipped",
            category="subplot",
        )

    plans = parsed.parsed_json.get("plans", [])
    if not isinstance(plans, list):
        return CheckResult(
            check_id="plot_thread_target_reasonable",
            passed=True,
            detail="No plans to check",
            category="subplot",
        )

    issues = []
    for plan in plans:
        if not isinstance(plan, dict):
            continue
        thread = plan.get("thread", "?")
        target = plan.get("target_turns")
        if target is None:
            continue
        target = int(target) if isinstance(target, (int, float)) else 0

        if target == 1:
            issues.append(f"'{thread}' has target_turns=1 (single-turn threads should be consequences, not plot threads)")

        # If resolving, skip target check
        if plan.get("status") == "resolved":
            continue

        scenario_turns = len(scenario.turns)
        if scenario_turns > 0 and target > scenario_turns * 10:
            issues.append(f"'{thread}' has target_turns={target} in a {scenario_turns}-turn scenario")

    if issues:
        return CheckResult(
            check_id="plot_thread_target_reasonable",
            passed=False,
            detail="; ".join(issues),
            category="subplot",
        )
    return CheckResult(
        check_id="plot_thread_target_reasonable",
        passed=True,
        detail="Plot thread target_turns values are reasonable",
        category="subplot",
    )


def plot_thread_not_orphaned(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check for plot threads that exceed their target_turns without resolution.

    Run this on later turns. If a thread's target_turns has elapsed and it's
    still active, the model failed to follow through on its own plan.
    """
    overdue = []
    for thread_id, thread in state.plot_threads.items():
        if thread.get("status") == "resolved":
            continue
        target = thread.get("target_turns", 0)
        created = thread.get("created_turn", 0)
        if target > 0 and created > 0:
            deadline = created + target
            if state.turn_number > deadline:
                overdue_by = state.turn_number - deadline
                overdue.append(f"'{thread_id}' target was {target} turns (created turn {created}), now {overdue_by} turns overdue")

    if overdue:
        return CheckResult(
            check_id="plot_thread_not_orphaned",
            passed=False,
            detail="; ".join(overdue),
            category="subplot",
        )
    return CheckResult(
        check_id="plot_thread_not_orphaned",
        passed=True,
        detail=f"{len(state.plot_threads)} thread(s) tracked, none overdue",
        category="subplot",
    )


# ── Chapter checks ───────────────────────────────────────────────────────


def chapter_fields_valid(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Validate chapter_plan tool call fields."""
    if parsed.parsed_json.get("tool_call") != "chapter_plan":
        return CheckResult(
            check_id="chapter_fields_valid",
            passed=True,
            detail="Not a chapter_plan tool call, skipped",
            category="subplot",
        )

    action = parsed.parsed_json.get("action", "")
    if action not in ("create", "update", "advance_scene", "resolve", "close"):
        return CheckResult(
            check_id="chapter_fields_valid",
            passed=False,
            detail=f"chapter_plan action '{action}' not valid",
            category="subplot",
        )

    chapter = parsed.parsed_json.get("chapter", {})
    issues = []

    if action == "create":
        if not isinstance(chapter, dict):
            issues.append("'chapter' must be a dict for create action")
        else:
            slug = chapter.get("slug", "")
            if not slug or not _slug_valid(slug):
                issues.append(f"Invalid chapter slug: '{slug}'")
            scenes = chapter.get("scenes", [])
            if not isinstance(scenes, list) or not scenes:
                issues.append("Chapter must have non-empty 'scenes' list")
            elif len(scenes) > 20:
                issues.append(f"Chapter scenes exceeds 20 ({len(scenes)})")
            title = chapter.get("title", "")
            if isinstance(title, str) and len(title) > 120:
                issues.append(f"Chapter title exceeds 120 chars ({len(title)})")

    elif action in ("advance_scene", "resolve"):
        # chapter can be a string slug or dict
        if isinstance(chapter, str):
            if not _slug_valid(chapter):
                issues.append(f"Invalid chapter slug: '{chapter}'")
        elif isinstance(chapter, dict):
            slug = chapter.get("slug", "")
            if not slug:
                issues.append("Chapter dict missing 'slug'")

    if issues:
        return CheckResult(
            check_id="chapter_fields_valid",
            passed=False,
            detail="; ".join(issues),
            category="subplot",
        )
    return CheckResult(
        check_id="chapter_fields_valid",
        passed=True,
        detail=f"Valid chapter_plan ({action})",
        category="subplot",
    )


def chapter_scene_progression(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that chapter scene advancement follows the declared scene order.

    If a chapter has scenes [a, b, c] and current_scene is 'a', advancing
    to 'c' (skipping 'b') is a violation.
    """
    if parsed.parsed_json.get("tool_call") != "chapter_plan":
        return CheckResult(
            check_id="chapter_scene_progression",
            passed=True,
            detail="Not a chapter_plan tool call, skipped",
            category="subplot",
        )

    if parsed.parsed_json.get("action") != "advance_scene":
        return CheckResult(
            check_id="chapter_scene_progression",
            passed=True,
            detail="Not an advance_scene action, skipped",
            category="subplot",
        )

    chapter_ref = parsed.parsed_json.get("chapter", "")
    to_scene = parsed.parsed_json.get("to_scene", "")
    slug = chapter_ref if isinstance(chapter_ref, str) else chapter_ref.get("slug", "") if isinstance(chapter_ref, dict) else ""

    if not slug or slug not in state.chapters:
        return CheckResult(
            check_id="chapter_scene_progression",
            passed=True,
            detail=f"Chapter '{slug}' not tracked yet, cannot verify progression",
            category="subplot",
        )

    chapter = state.chapters[slug]
    scenes = chapter.get("scenes", [])
    current = chapter.get("current_scene", "")

    if not scenes or not current or not to_scene:
        return CheckResult(
            check_id="chapter_scene_progression",
            passed=True,
            detail="Insufficient scene data to verify progression",
            category="subplot",
        )

    try:
        current_idx = scenes.index(current)
        target_idx = scenes.index(to_scene)
    except ValueError:
        return CheckResult(
            check_id="chapter_scene_progression",
            passed=True,
            detail=f"Scene '{to_scene}' or '{current}' not in scenes list",
            category="subplot",
        )

    if target_idx == current_idx + 1:
        return CheckResult(
            check_id="chapter_scene_progression",
            passed=True,
            detail=f"Scene advances correctly: '{current}' -> '{to_scene}'",
            category="subplot",
        )
    elif target_idx <= current_idx:
        return CheckResult(
            check_id="chapter_scene_progression",
            passed=False,
            detail=f"Scene goes backward: '{current}' (idx {current_idx}) -> '{to_scene}' (idx {target_idx})",
            category="subplot",
        )
    else:
        skipped = scenes[current_idx + 1:target_idx]
        return CheckResult(
            check_id="chapter_scene_progression",
            passed=False,
            detail=f"Skipped scenes: {skipped} between '{current}' and '{to_scene}'",
            category="subplot",
        )


# ── Consequence checks ───────────────────────────────────────────────────


def consequence_fields_valid(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Validate consequence_log tool call fields."""
    if parsed.parsed_json.get("tool_call") != "consequence_log":
        return CheckResult(
            check_id="consequence_fields_valid",
            passed=True,
            detail="Not a consequence_log tool call, skipped",
            category="subplot",
        )

    issues = []

    # Check adds
    adds = parsed.parsed_json.get("add", [])
    if isinstance(adds, dict):
        adds = [adds]
    if isinstance(adds, list):
        for i, item in enumerate(adds):
            if not isinstance(item, dict):
                issues.append(f"add[{i}] not a dict")
                continue
            trigger = item.get("trigger", "")
            if not trigger:
                issues.append(f"add[{i}] missing 'trigger'")
            consequence = item.get("consequence", "")
            if not consequence:
                issues.append(f"add[{i}] missing 'consequence'")

            severity = item.get("severity", "")
            if severity and severity not in ("low", "moderate", "high", "critical"):
                issues.append(f"add[{i}] severity '{severity}' not valid")

            expires = item.get("expires_turns")
            if expires is not None:
                if not isinstance(expires, (int, float)):
                    issues.append(f"add[{i}] expires_turns not numeric")
                elif int(expires) < 0:
                    issues.append(f"add[{i}] expires_turns negative")

            # Length limits
            if isinstance(trigger, str) and len(trigger) > 240:
                issues.append(f"add[{i}] trigger exceeds 240 chars")
            if isinstance(consequence, str) and len(consequence) > 300:
                issues.append(f"add[{i}] consequence exceeds 300 chars")

    # Check resolves
    resolves = parsed.parsed_json.get("resolve", [])
    if isinstance(resolves, dict):
        resolves = [resolves]
    if isinstance(resolves, list):
        for i, item in enumerate(resolves):
            if isinstance(item, dict):
                if not item.get("id"):
                    issues.append(f"resolve[{i}] missing 'id'")

    if issues:
        return CheckResult(
            check_id="consequence_fields_valid",
            passed=False,
            detail="; ".join(issues),
            category="subplot",
        )
    return CheckResult(
        check_id="consequence_fields_valid",
        passed=True,
        detail="Valid consequence_log fields",
        category="subplot",
    )


def consequence_severity_proportional(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that consequence severity matches the trigger's magnitude.

    A "critical" consequence from a minor action is disproportionate.
    """
    if parsed.parsed_json.get("tool_call") != "consequence_log":
        return CheckResult(
            check_id="consequence_severity_proportional",
            passed=True,
            detail="Not a consequence_log tool call, skipped",
            category="subplot",
        )

    adds = parsed.parsed_json.get("add", [])
    if isinstance(adds, dict):
        adds = [adds]
    if not isinstance(adds, list):
        return CheckResult(
            check_id="consequence_severity_proportional",
            passed=True,
            detail="No consequences added",
            category="subplot",
        )

    critical_count = sum(
        1 for a in adds
        if isinstance(a, dict) and a.get("severity") == "critical"
    )
    high_count = sum(
        1 for a in adds
        if isinstance(a, dict) and a.get("severity") == "high"
    )

    # Flag if more than one critical or high consequence in a single turn
    if critical_count > 1:
        return CheckResult(
            check_id="consequence_severity_proportional",
            passed=False,
            detail=f"{critical_count} critical consequences in a single turn is likely disproportionate",
            category="subplot",
        )

    if critical_count + high_count > 2:
        return CheckResult(
            check_id="consequence_severity_proportional",
            passed=False,
            detail=f"{critical_count} critical + {high_count} high consequences in a single turn",
            category="subplot",
        )

    return CheckResult(
        check_id="consequence_severity_proportional",
        passed=True,
        detail=f"Consequence severity appears proportional ({len(adds)} added)",
        category="subplot",
    )
