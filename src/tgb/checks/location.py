"""Location checks: location_coherent (5-field room check)."""

from __future__ import annotations

from typing import Any

from tgb.checks.base import CheckResult
from tgb.config import Scenario, TurnSpec
from tgb.prompt_builder import AccumulatedState
from tgb.response_parser import ParsedResponse

ROOM_FIELDS = ["location", "room_title", "room_summary", "room_description", "exits"]


def check_location_coherent(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that all 5 room fields are present in player_state_update on movement.

    Params:
        expect_move: bool — if True, movement is expected and all fields must be present
    """
    expect_move = params.get("expect_move", False)
    player_update = parsed.parsed_json.get("player_state_update")

    if not isinstance(player_update, dict):
        if expect_move:
            return CheckResult(
                check_id="location_coherent",
                passed=False,
                detail="Movement expected but no player_state_update",
                category="location",
            )
        return CheckResult(
            check_id="location_coherent",
            passed=True,
            detail="No player_state_update (no movement expected)",
            category="location",
        )

    # Detect if location changed
    location_changed = "location" in player_update or "room_title" in player_update
    if not location_changed and not expect_move:
        return CheckResult(
            check_id="location_coherent",
            passed=True,
            detail="No location change detected",
            category="location",
        )

    # Check all 5 fields present
    missing = [f for f in ROOM_FIELDS if f not in player_update]
    if missing:
        return CheckResult(
            check_id="location_coherent",
            passed=False,
            detail=f"Location change but missing room fields: {missing}",
            category="location",
        )

    # Validate field types
    type_issues = []
    if not isinstance(player_update.get("location"), str):
        type_issues.append("location not a string")
    if not isinstance(player_update.get("room_title"), str):
        type_issues.append("room_title not a string")
    if not isinstance(player_update.get("room_summary"), str):
        type_issues.append("room_summary not a string")
    if not isinstance(player_update.get("room_description"), str):
        type_issues.append("room_description not a string")
    exits = player_update.get("exits")
    if not isinstance(exits, (list, str)):
        type_issues.append("exits not a list or string")

    if type_issues:
        return CheckResult(
            check_id="location_coherent",
            passed=False,
            detail=f"Room field type issues: {'; '.join(type_issues)}",
            category="location",
        )

    return CheckResult(
        check_id="location_coherent",
        passed=True,
        detail=f"All 5 room fields present and valid (location: {player_update['location']})",
        category="location",
    )
