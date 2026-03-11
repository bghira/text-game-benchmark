"""Calendar checks — validates calendar_update against the engine contract.

The engine's calendar contract:
- calendar_update is a dict with optional "add" and "remove" keys
- "add" is a list of event dicts, each requiring: name, time_remaining, time_unit
- "remove" is a list of event name strings to remove
- time_unit must be "hours" or "days"
- time_remaining must be a positive integer
- known_by is an optional list of character names
- target_player / target_players are optional player references
- Engine normalizes time_remaining/time_unit into fire_day/fire_hour:
  fire_day >= 1, fire_hour in 0-23
"""

from __future__ import annotations

from typing import Any

from tgb.checks.base import CheckResult
from tgb.checks.limits import CALENDAR_NAME_MAX_CHARS
from tgb.config import Scenario, TurnSpec
from tgb.prompt_builder import AccumulatedState
from tgb.response_parser import ParsedResponse

VALID_TIME_UNITS = {"hours", "days"}
EVENT_REQUIRED_FIELDS = {"name", "time_remaining", "time_unit"}
EVENT_OPTIONAL_FIELDS = {
    "description", "known_by", "target_player", "target_players",
    "fire_day", "fire_hour",
}
EVENT_VALID_FIELDS = EVENT_REQUIRED_FIELDS | EVENT_OPTIONAL_FIELDS

# Legacy fields that the engine auto-normalizes — models should not emit these
LEGACY_FIELDS = {
    "status", "_status", "_hours_until", "_days_until",
    "hours_remaining", "days_remaining", "created_day", "created_hour",
}


def check_calendar_update_valid(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Validate calendar_update structure and field types."""
    cal = parsed.parsed_json.get("calendar_update")
    if cal is None:
        return CheckResult(
            check_id="calendar_update_valid",
            passed=True,
            detail="No calendar_update",
            category="calendar",
        )

    if not isinstance(cal, dict):
        return CheckResult(
            check_id="calendar_update_valid",
            passed=False,
            detail=f"calendar_update is {type(cal).__name__}, expected dict",
            category="calendar",
        )

    issues: list[str] = []

    # Validate top-level keys
    valid_top = {"add", "remove"}
    unknown_top = set(cal.keys()) - valid_top
    if unknown_top:
        issues.append(f"unknown top-level keys: {sorted(unknown_top)}")

    # Validate "add" entries
    add_list = cal.get("add")
    if add_list is not None:
        if not isinstance(add_list, list):
            issues.append(f"'add' is {type(add_list).__name__}, expected list")
        else:
            for i, event in enumerate(add_list):
                if not isinstance(event, dict):
                    issues.append(f"add[{i}] is {type(event).__name__}, expected dict")
                    continue

                # Required fields
                missing = EVENT_REQUIRED_FIELDS - set(event.keys())
                if missing:
                    issues.append(f"add[{i}] missing required fields: {sorted(missing)}")

                # name validation
                name = event.get("name")
                if name is not None:
                    if not isinstance(name, str) or not name.strip():
                        issues.append(f"add[{i}].name must be a non-empty string")
                    elif len(name) > CALENDAR_NAME_MAX_CHARS:
                        issues.append(f"add[{i}].name too long ({len(name)} > {CALENDAR_NAME_MAX_CHARS})")

                # time_remaining validation
                tr = event.get("time_remaining")
                if tr is not None:
                    if isinstance(tr, bool) or not isinstance(tr, (int, float)):
                        issues.append(f"add[{i}].time_remaining must be a positive integer")
                    elif isinstance(tr, float) and tr != int(tr):
                        issues.append(f"add[{i}].time_remaining must be an integer, got {tr}")
                    elif int(tr) < 1:
                        issues.append(f"add[{i}].time_remaining must be >= 1")

                # time_unit validation
                tu = event.get("time_unit")
                if tu is not None and tu not in VALID_TIME_UNITS:
                    issues.append(f"add[{i}].time_unit='{tu}' not in {VALID_TIME_UNITS}")

                # known_by validation
                kb = event.get("known_by")
                if kb is not None:
                    if not isinstance(kb, list):
                        issues.append(f"add[{i}].known_by must be a list")
                    elif not all(isinstance(k, str) for k in kb):
                        issues.append(f"add[{i}].known_by entries must be strings")

                # Reject unknown keys
                unknown_keys = set(event.keys()) - EVENT_VALID_FIELDS
                # Separate legacy fields for clearer messaging
                legacy_present = unknown_keys & LEGACY_FIELDS
                other_unknown = unknown_keys - LEGACY_FIELDS
                if legacy_present:
                    issues.append(f"add[{i}] has legacy fields {sorted(legacy_present)} — use time_remaining/time_unit instead")
                if other_unknown:
                    issues.append(f"add[{i}] has unknown fields {sorted(other_unknown)}")

    # Validate "remove" entries
    remove_list = cal.get("remove")
    if remove_list is not None:
        if not isinstance(remove_list, list):
            issues.append(f"'remove' is {type(remove_list).__name__}, expected list")
        else:
            for i, name in enumerate(remove_list):
                if not isinstance(name, str) or not name.strip():
                    issues.append(f"remove[{i}] must be a non-empty string")

    if issues:
        return CheckResult(
            check_id="calendar_update_valid",
            passed=False,
            detail="; ".join(issues),
            category="calendar",
        )
    return CheckResult(
        check_id="calendar_update_valid",
        passed=True,
        detail="calendar_update valid",
        category="calendar",
    )


def check_calendar_no_legacy_fields(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that calendar events don't use legacy fields the engine auto-normalizes."""
    cal = parsed.parsed_json.get("calendar_update")
    if not isinstance(cal, dict):
        return CheckResult(
            check_id="calendar_no_legacy_fields",
            passed=True,
            detail="No calendar_update",
            category="calendar",
        )

    add_list = cal.get("add")
    if not isinstance(add_list, list):
        return CheckResult(
            check_id="calendar_no_legacy_fields",
            passed=True,
            detail="No add list",
            category="calendar",
        )

    all_legacy: list[str] = []
    for i, event in enumerate(add_list):
        if not isinstance(event, dict):
            continue
        found = set(event.keys()) & LEGACY_FIELDS
        if found:
            all_legacy.extend(f"add[{i}].{f}" for f in sorted(found))

    if all_legacy:
        return CheckResult(
            check_id="calendar_no_legacy_fields",
            passed=False,
            detail=f"Legacy calendar fields present: {all_legacy}",
            category="calendar",
        )
    return CheckResult(
        check_id="calendar_no_legacy_fields",
        passed=True,
        detail="No legacy calendar fields",
        category="calendar",
    )


def check_calendar_fire_range(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Validate fire_day/fire_hour ranges in calendar add events.

    Engine enforces: fire_day >= 1, fire_hour in 0-23.
    """
    cal = parsed.parsed_json.get("calendar_update")
    if not isinstance(cal, dict):
        return CheckResult(
            check_id="calendar_fire_range",
            passed=True,
            detail="No calendar_update",
            category="calendar",
        )

    add_list = cal.get("add")
    if not isinstance(add_list, list):
        return CheckResult(
            check_id="calendar_fire_range",
            passed=True,
            detail="No add list",
            category="calendar",
        )

    issues: list[str] = []
    for i, event in enumerate(add_list):
        if not isinstance(event, dict):
            continue

        fire_day = event.get("fire_day")
        if fire_day is not None:
            if isinstance(fire_day, bool) or not isinstance(fire_day, (int, float)):
                issues.append(f"add[{i}].fire_day must be an integer")
            elif int(fire_day) < 1:
                issues.append(f"add[{i}].fire_day={int(fire_day)} must be >= 1")

        fire_hour = event.get("fire_hour")
        if fire_hour is not None:
            if isinstance(fire_hour, bool) or not isinstance(fire_hour, (int, float)):
                issues.append(f"add[{i}].fire_hour must be an integer")
            elif not (0 <= int(fire_hour) <= 23):
                issues.append(f"add[{i}].fire_hour={int(fire_hour)} must be in 0-23")

    if issues:
        return CheckResult(
            check_id="calendar_fire_range",
            passed=False,
            detail="; ".join(issues),
            category="calendar",
        )
    return CheckResult(
        check_id="calendar_fire_range",
        passed=True,
        detail="Calendar fire_day/fire_hour ranges valid",
        category="calendar",
    )
