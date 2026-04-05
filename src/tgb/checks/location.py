"""Location checks: location_coherent (5-field room check), location_updates_valid."""

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


import re

LOCATION_SLUG_PATTERN = re.compile(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$")
VALID_PRIORITY_VALUES = {"critical", "scene", "low"}

# Engine normalizes these aliases to canonical values
PRIORITY_ALIASES: dict[str, str] = {
    "always": "critical", "sticky": "critical", "persistent": "critical",
    "active": "scene", "local": "scene",
    "minor": "low", "ephemeral": "low", "temporary": "low",
}


def check_location_updates_valid(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Validate location_updates field structure.

    Engine expects:
    - Dict keyed by stable location slugs (e.g. "hotel-lobby")
    - Each value is a dict of location facts
    - Facts can be plain values or priority-wrapped: {"value": "...", "priority": "critical"}
    - null/remove/delete values signal deletion
    """
    updates = parsed.parsed_json.get("location_updates")
    if updates is None:
        return CheckResult(
            check_id="location_updates_valid",
            passed=True,
            detail="No location_updates field, skipped",
            category="location",
        )

    if not isinstance(updates, dict):
        return CheckResult(
            check_id="location_updates_valid",
            passed=False,
            detail=f"location_updates must be a dict, got {type(updates).__name__}",
            category="location",
        )

    issues: list[str] = []
    for slug, loc_data in updates.items():
        if not isinstance(slug, str):
            issues.append(f"location key {slug!r} is not a string")
            continue

        # Validate slug format (lowercase-hyphenated)
        if not LOCATION_SLUG_PATTERN.match(slug):
            issues.append(f"location slug '{slug}' is not lowercase-hyphenated format")

        # null means deletion — ok
        if loc_data is None:
            continue

        if not isinstance(loc_data, dict):
            issues.append(f"location_updates['{slug}'] must be a dict, got {type(loc_data).__name__}")
            continue

        # Validate priority-wrapped fields
        for fact_key, fact_val in loc_data.items():
            if fact_key == "_fact_priorities_key":
                continue  # Engine internal
            if isinstance(fact_val, dict):
                # Priority-wrapped value
                if "value" not in fact_val:
                    issues.append(
                        f"location_updates['{slug}'].{fact_key} is a dict but missing 'value' key"
                    )
                priority = fact_val.get("priority")
                if priority is not None:
                    # Accept canonical values and known aliases
                    if priority not in VALID_PRIORITY_VALUES and priority not in PRIORITY_ALIASES:
                        issues.append(
                            f"location_updates['{slug}'].{fact_key} priority '{priority}' "
                            f"not in {VALID_PRIORITY_VALUES} or known aliases"
                        )

    if issues:
        return CheckResult(
            check_id="location_updates_valid",
            passed=False,
            detail="; ".join(issues[:5]),
            category="location",
        )
    return CheckResult(
        check_id="location_updates_valid",
        passed=True,
        detail=f"location_updates valid ({len(updates)} locations)",
        category="location",
    )
