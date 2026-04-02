"""State management checks: state_nested, state_null_prune, state_completed_value_prune, game_time_advanced, game_time_period_valid, state_update_required_fields, state_no_character_removal, inventory_changes_limit, summary_update_valid, party_status_valid."""

from __future__ import annotations

import re
from typing import Any

from tgb.checks.base import CheckResult
from tgb.checks.limits import MAX_INVENTORY_CHANGES_PER_TURN
from tgb.config import Scenario, TurnSpec
from tgb.prompt_builder import AccumulatedState
from tgb.response_parser import ParsedResponse

# Pattern for flat underscore-joined keys (e.g. guard_captain_mood)
FLAT_KEY_PATTERN = re.compile(r"^[a-z]+(_[a-z]+){2,}$")


def check_state_nested(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that state_update uses nested objects, not flat underscore-joined keys."""
    state_update = parsed.parsed_json.get("state_update")
    if not isinstance(state_update, dict) or not state_update:
        return CheckResult(
            check_id="state_nested",
            passed=True,
            detail="No state_update or empty",
            category="state_mgmt",
        )

    flat_keys = [k for k in state_update if FLAT_KEY_PATTERN.match(k)]
    if flat_keys:
        return CheckResult(
            check_id="state_nested",
            passed=False,
            detail=f"Flat underscore-joined keys: {flat_keys[:5]}",
            category="state_mgmt",
        )
    return CheckResult(
        check_id="state_nested",
        passed=True,
        detail="All state_update keys properly nested",
        category="state_mgmt",
    )


def check_state_null_prune(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that completed/resolved keys are pruned (set to null) in state_update.

    Params:
        expect_pruned: list of dot-path keys expected to be set to null
    """
    state_update = parsed.parsed_json.get("state_update")
    expect_pruned = params.get("expect_pruned", [])

    if not expect_pruned:
        return CheckResult(
            check_id="state_null_prune",
            passed=True,
            detail="No specific pruning expectations",
            category="state_mgmt",
        )

    if not isinstance(state_update, dict):
        return CheckResult(
            check_id="state_null_prune",
            passed=False,
            detail=f"state_update missing; expected pruning of {expect_pruned}",
            category="state_mgmt",
        )

    not_pruned = []
    for path in expect_pruned:
        parts = path.split(".")
        obj = state_update
        found = True
        for i, part in enumerate(parts):
            if not isinstance(obj, dict) or part not in obj:
                found = False
                break
            if i < len(parts) - 1:
                obj = obj[part]
            else:
                if obj[part] is not None:
                    not_pruned.append(path)

        if not found:
            # Key not in state_update at all — might be fine if it was already pruned
            # or was never set. Check if it's still in accumulated state.
            current = state.campaign_state
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    current = None
                    break
            if current is not None:
                not_pruned.append(f"{path} (still in state, not pruned)")

    if not_pruned:
        return CheckResult(
            check_id="state_null_prune",
            passed=False,
            detail=f"Keys not pruned: {not_pruned}",
            category="state_mgmt",
        )
    return CheckResult(
        check_id="state_null_prune",
        passed=True,
        detail=f"All {len(expect_pruned)} expected keys pruned",
        category="state_mgmt",
    )


# Engine auto-deletes keys set to these string values instead of storing them.
COMPLETED_VALUES = {
    "complete", "completed", "done", "resolved", "finished",
    "concluded", "vacated", "dispersed", "avoided", "departed",
}


def check_state_completed_value_prune(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that state_update doesn't set keys to completed-value strings.

    The engine auto-deletes keys set to values like 'completed', 'done',
    'resolved', etc. — it pops the key instead of storing the string.
    This check warns when a model emits such values, since the engine
    will delete the key rather than storing the string.
    """
    state_update = parsed.parsed_json.get("state_update")
    if not isinstance(state_update, dict) or not state_update:
        return CheckResult(
            check_id="state_completed_value_prune",
            passed=True,
            detail="No state_update",
            category="state_mgmt",
        )

    pruned_keys: list[str] = []
    _walk_for_completed(state_update, "", pruned_keys)

    if pruned_keys:
        return CheckResult(
            check_id="state_completed_value_prune",
            passed=False,
            detail=f"Keys set to completed-value strings (engine will delete instead of store): {pruned_keys[:5]}",
            category="state_mgmt",
        )
    return CheckResult(
        check_id="state_completed_value_prune",
        passed=True,
        detail="No completed-value strings in state_update",
        category="state_mgmt",
    )


def _walk_for_completed(obj: dict[str, Any], prefix: str, results: list[str]) -> None:
    """Recursively find keys set to completed-value strings."""
    for key, val in obj.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(val, str) and val.strip().lower() in COMPLETED_VALUES:
            results.append(path)
        elif isinstance(val, dict):
            _walk_for_completed(val, path, results)


def check_game_time_advanced(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that state_update advances game_time.

    The engine requires: 'Every turn, you MUST advance game_time in
    state_update by a plausible amount.'
    """
    if parsed.is_tool_call:
        return CheckResult(
            check_id="game_time_advanced",
            passed=True,
            detail="Tool call — game_time check skipped",
            category="state_mgmt",
        )

    state_update = parsed.parsed_json.get("state_update")
    if not isinstance(state_update, dict):
        return CheckResult(
            check_id="game_time_advanced",
            passed=False,
            detail="state_update missing — game_time must be advanced every turn",
            category="state_mgmt",
        )

    game_time = state_update.get("game_time")
    if game_time is None:
        return CheckResult(
            check_id="game_time_advanced",
            passed=False,
            detail="game_time not in state_update — must advance every turn",
            category="state_mgmt",
        )

    if not isinstance(game_time, dict):
        return CheckResult(
            check_id="game_time_advanced",
            passed=False,
            detail=f"game_time is {type(game_time).__name__}, expected dict",
            category="state_mgmt",
        )

    return CheckResult(
        check_id="game_time_advanced",
        passed=True,
        detail=f"game_time advanced: {game_time}",
        category="state_mgmt",
    )


def check_game_time_range_valid(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that game_time fields are in valid ranges.

    Engine enforces: day >= 1, hour 0-23, minute 0-59.
    """
    if parsed.is_tool_call:
        return CheckResult(
            check_id="game_time_range_valid",
            passed=True,
            detail="Tool call — range check skipped",
            category="state_mgmt",
        )

    state_update = parsed.parsed_json.get("state_update")
    if not isinstance(state_update, dict):
        return CheckResult(
            check_id="game_time_range_valid",
            passed=True,
            detail="No state_update",
            category="state_mgmt",
        )

    game_time = state_update.get("game_time")
    if not isinstance(game_time, dict):
        return CheckResult(
            check_id="game_time_range_valid",
            passed=True,
            detail="No game_time dict in state_update",
            category="state_mgmt",
        )

    issues: list[str] = []

    day = game_time.get("day")
    if day is not None:
        if not isinstance(day, (int, float)) or isinstance(day, bool):
            issues.append(f"day must be an integer, got {type(day).__name__}")
        elif int(day) < 1:
            issues.append(f"day={int(day)} is less than 1")

    hour = game_time.get("hour")
    if hour is not None:
        if not isinstance(hour, (int, float)) or isinstance(hour, bool):
            issues.append(f"hour must be an integer, got {type(hour).__name__}")
        elif int(hour) < 0 or int(hour) > 23:
            issues.append(f"hour={int(hour)} out of range [0, 23]")

    minute = game_time.get("minute")
    if minute is not None:
        if not isinstance(minute, (int, float)) or isinstance(minute, bool):
            issues.append(f"minute must be an integer, got {type(minute).__name__}")
        elif int(minute) < 0 or int(minute) > 59:
            issues.append(f"minute={int(minute)} out of range [0, 59]")

    if issues:
        return CheckResult(
            check_id="game_time_range_valid",
            passed=False,
            detail="; ".join(issues),
            category="state_mgmt",
        )
    return CheckResult(
        check_id="game_time_range_valid",
        passed=True,
        detail=f"game_time ranges valid: {game_time}",
        category="state_mgmt",
    )


def check_state_no_character_removal(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that state_update does not null character roster entries.

    The engine rule: 'state_update is NEVER a roster-deletion mechanism.
    Do NOT remove characters via state_update. Use character_updates.'
    Flags any top-level key in state_update that matches a known character
    slug being set to null.
    """
    state_update = parsed.parsed_json.get("state_update")
    if not isinstance(state_update, dict) or not state_update:
        return CheckResult(
            check_id="state_no_character_removal",
            passed=True,
            detail="No state_update",
            category="state_mgmt",
        )

    character_slugs = set(state.characters.keys())
    if not character_slugs:
        return CheckResult(
            check_id="state_no_character_removal",
            passed=True,
            detail="No known characters to check against",
            category="state_mgmt",
        )

    removed = [k for k in state_update if k in character_slugs and state_update[k] is None]
    if removed:
        return CheckResult(
            check_id="state_no_character_removal",
            passed=False,
            detail=f"state_update nulls character slugs {removed} — use character_updates instead",
            category="state_mgmt",
        )
    return CheckResult(
        check_id="state_no_character_removal",
        passed=True,
        detail="No character removal via state_update",
        category="state_mgmt",
    )


def check_inventory_changes_limit(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that inventory changes per turn don't exceed the engine limit.

    The engine enforces MAX_INVENTORY_CHANGES_PER_TURN (10) combining both
    inventory_add and inventory_remove arrays.
    """
    player_update = parsed.parsed_json.get("player_state_update")
    if not isinstance(player_update, dict):
        return CheckResult(
            check_id="inventory_changes_limit",
            passed=True,
            detail="No player_state_update",
            category="state_mgmt",
        )

    max_changes = params.get("max", MAX_INVENTORY_CHANGES_PER_TURN)

    adds = player_update.get("inventory_add", [])
    removes = player_update.get("inventory_remove", [])
    add_count = len(adds) if isinstance(adds, list) else 0
    remove_count = len(removes) if isinstance(removes, list) else 0
    total = add_count + remove_count

    if total > max_changes:
        return CheckResult(
            check_id="inventory_changes_limit",
            passed=False,
            detail=f"{total} inventory changes ({add_count} add + {remove_count} remove) exceeds max {max_changes}",
            category="state_mgmt",
        )
    return CheckResult(
        check_id="inventory_changes_limit",
        passed=True,
        detail=f"{total} inventory changes" if total else "No inventory changes",
        category="state_mgmt",
    )


# ── game_time period validation ─────────────────────────────────

VALID_PERIODS = {"morning", "afternoon", "evening", "night"}


def check_game_time_period_valid(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that game_time.period is one of: morning, afternoon, evening, night.

    Engine mapping: 5-11=morning, 12-16=afternoon, 17-20=evening, 21-4=night.
    """
    if parsed.is_tool_call:
        return CheckResult(
            check_id="game_time_period_valid",
            passed=True,
            detail="Tool call — period check skipped",
            category="state_mgmt",
        )

    state_update = parsed.parsed_json.get("state_update")
    if not isinstance(state_update, dict):
        return CheckResult(
            check_id="game_time_period_valid",
            passed=True,
            detail="No state_update",
            category="state_mgmt",
        )

    game_time = state_update.get("game_time")
    if not isinstance(game_time, dict):
        return CheckResult(
            check_id="game_time_period_valid",
            passed=True,
            detail="No game_time dict in state_update",
            category="state_mgmt",
        )

    period = game_time.get("period")
    if period is None:
        return CheckResult(
            check_id="game_time_period_valid",
            passed=True,
            detail="No period field in game_time",
            category="state_mgmt",
        )

    if not isinstance(period, str) or period not in VALID_PERIODS:
        return CheckResult(
            check_id="game_time_period_valid",
            passed=False,
            detail=f"game_time.period '{period}' not in {sorted(VALID_PERIODS)}",
            category="state_mgmt",
        )

    return CheckResult(
        check_id="game_time_period_valid",
        passed=True,
        detail=f"game_time.period='{period}' valid",
        category="state_mgmt",
    )


# ── state_update required fields ─────────────────────────────────

STATE_UPDATE_REQUIRED = {"game_time", "current_chapter", "current_scene"}


def check_state_update_required_fields(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that state_update includes required fields.

    Engine rule: 'state_update is required and must include "game_time",
    "current_chapter", and "current_scene" explicitly.'
    """
    if parsed.is_tool_call:
        return CheckResult(
            check_id="state_update_required_fields",
            passed=True,
            detail="Tool call — required fields check skipped",
            category="state_mgmt",
        )

    state_update = parsed.parsed_json.get("state_update")
    if not isinstance(state_update, dict):
        return CheckResult(
            check_id="state_update_required_fields",
            passed=False,
            detail="state_update missing — must include game_time, current_chapter, current_scene",
            category="state_mgmt",
        )

    required = set(params.get("required", list(STATE_UPDATE_REQUIRED)))
    missing = [k for k in sorted(required) if k not in state_update]
    if missing:
        return CheckResult(
            check_id="state_update_required_fields",
            passed=False,
            detail=f"state_update missing required fields: {missing}",
            category="state_mgmt",
        )

    return CheckResult(
        check_id="state_update_required_fields",
        passed=True,
        detail=f"All {len(required)} required state_update fields present",
        category="state_mgmt",
    )


# ── summary_update validation ───────────────────────────────────


def check_summary_update_valid(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that summary_update is a concise string (1-2 sentences of lasting changes).

    Engine rule: 'summary_update: string (one or two sentences of lasting changes)'.
    """
    if parsed.is_tool_call:
        return CheckResult(
            check_id="summary_update_valid",
            passed=True,
            detail="Tool call — summary_update check skipped",
            category="state_mgmt",
        )

    summary = parsed.parsed_json.get("summary_update")
    if summary is None:
        return CheckResult(
            check_id="summary_update_valid",
            passed=True,
            detail="No summary_update present",
            category="state_mgmt",
        )

    if not isinstance(summary, str):
        return CheckResult(
            check_id="summary_update_valid",
            passed=False,
            detail=f"summary_update is {type(summary).__name__}, expected str",
            category="state_mgmt",
        )

    stripped = summary.strip()
    if not stripped:
        return CheckResult(
            check_id="summary_update_valid",
            passed=False,
            detail="summary_update is empty",
            category="state_mgmt",
        )

    max_chars = params.get("max_chars", 500)
    if len(stripped) > max_chars:
        return CheckResult(
            check_id="summary_update_valid",
            passed=False,
            detail=f"summary_update is {len(stripped)} chars, max {max_chars}",
            category="state_mgmt",
        )

    return CheckResult(
        check_id="summary_update_valid",
        passed=True,
        detail=f"summary_update valid ({len(stripped)} chars)",
        category="state_mgmt",
    )


# ── party_status validation ─────────────────────────────────────

VALID_PARTY_STATUSES = {"main_party", "new_path"}


def check_party_status_valid(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that party_status in player_state_update is 'main_party' or 'new_path'.

    Engine rule: 'NEVER change party_status away from main_party unless the
    player EXPLICITLY requests to split off or go solo.'
    """
    player_update = parsed.parsed_json.get("player_state_update")
    if not isinstance(player_update, dict):
        return CheckResult(
            check_id="party_status_valid",
            passed=True,
            detail="No player_state_update",
            category="state_mgmt",
        )

    status = player_update.get("party_status")
    if status is None:
        return CheckResult(
            check_id="party_status_valid",
            passed=True,
            detail="No party_status change",
            category="state_mgmt",
        )

    if not isinstance(status, str) or status not in VALID_PARTY_STATUSES:
        return CheckResult(
            check_id="party_status_valid",
            passed=False,
            detail=f"party_status '{status}' not in {sorted(VALID_PARTY_STATUSES)}",
            category="state_mgmt",
        )

    return CheckResult(
        check_id="party_status_valid",
        passed=True,
        detail=f"party_status='{status}' valid",
        category="state_mgmt",
    )


STORY_PROGRESSION_TARGETS = {"hold", "next-scene", "next-chapter"}


def check_story_progression_valid(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Validate story_progression field structure.

    Engine expects (on_rails campaigns only):
    - advance: bool
    - target: "hold" | "next-scene" | "next-chapter"
    - reason: string
    """
    prog = parsed.parsed_json.get("story_progression")
    if prog is None:
        return CheckResult(
            check_id="story_progression_valid",
            passed=True,
            detail="No story_progression field, skipped",
            category="state_mgmt",
        )

    if not isinstance(prog, dict):
        return CheckResult(
            check_id="story_progression_valid",
            passed=False,
            detail=f"story_progression must be a dict, got {type(prog).__name__}",
            category="state_mgmt",
        )

    issues: list[str] = []

    # advance validation
    advance = prog.get("advance")
    if advance is not None and not isinstance(advance, bool):
        issues.append(f"'advance' must be a bool, got {type(advance).__name__}")

    # target validation
    target = prog.get("target")
    if target is not None:
        if not isinstance(target, str):
            issues.append(f"'target' must be a string, got {type(target).__name__}")
        else:
            # Normalize underscores to hyphens (engine does this)
            normalized = target.strip().lower().replace("_", "-")
            if normalized not in STORY_PROGRESSION_TARGETS:
                issues.append(
                    f"'target' '{target}' not in {sorted(STORY_PROGRESSION_TARGETS)}"
                )

    # reason validation
    reason = prog.get("reason")
    if reason is not None and not isinstance(reason, str):
        issues.append(f"'reason' must be a string, got {type(reason).__name__}")

    if issues:
        return CheckResult(
            check_id="story_progression_valid",
            passed=False,
            detail="; ".join(issues),
            category="state_mgmt",
        )
    return CheckResult(
        check_id="story_progression_valid",
        passed=True,
        detail="story_progression valid",
        category="state_mgmt",
    )
