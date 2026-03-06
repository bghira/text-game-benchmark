"""State management checks: state_nested, state_null_prune."""

from __future__ import annotations

import re
from typing import Any

from tgb.checks.base import CheckResult
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
