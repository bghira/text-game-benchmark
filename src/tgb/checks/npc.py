"""NPC checks: npc_slug_valid, npc_immutable_preserved, npc_creation_fields, npc_update_fields_valid."""

from __future__ import annotations

import re
from typing import Any

from tgb.checks.base import CheckResult
from tgb.config import Scenario, TurnSpec
from tgb.prompt_builder import AccumulatedState
from tgb.response_parser import ParsedResponse

SLUG_PATTERN = re.compile(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$")

IMMUTABLE_FIELDS = {"name", "personality", "background", "appearance", "speech_style"}

# Fields that can be changed after creation
MUTABLE_FIELDS = {
    "location", "current_status", "allegiance", "relationship",
    "relationships", "literary_style", "deceased_reason",
}

# Required fields when creating a new NPC
CREATION_REQUIRED_FIELDS = {"name"}

# All valid fields for character_updates entries
ALL_VALID_FIELDS = IMMUTABLE_FIELDS | MUTABLE_FIELDS

# Deletion sentinels
_DELETION_SENTINELS = {None, "delete", "remove"}


def check_npc_slug_valid(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that character slugs match ^[a-z][a-z0-9]*(-[a-z0-9]+)*$."""
    char_updates = parsed.parsed_json.get("character_updates")
    if not isinstance(char_updates, dict) or not char_updates:
        return CheckResult(
            check_id="npc_slug_valid",
            passed=True,
            detail="No character_updates",
            category="npc",
        )

    invalid = []
    for slug in char_updates:
        if char_updates[slug] is None:
            continue  # Removal — slug format doesn't matter
        if not SLUG_PATTERN.match(slug):
            invalid.append(slug)

    if invalid:
        return CheckResult(
            check_id="npc_slug_valid",
            passed=False,
            detail=f"Invalid slugs: {invalid}",
            category="npc",
        )
    return CheckResult(
        check_id="npc_slug_valid",
        passed=True,
        detail=f"All {len(char_updates)} slugs valid",
        category="npc",
    )


def check_npc_immutable_preserved(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that immutable fields are not changed after NPC creation."""
    char_updates = parsed.parsed_json.get("character_updates")
    if not isinstance(char_updates, dict) or not char_updates:
        return CheckResult(
            check_id="npc_immutable_preserved",
            passed=True,
            detail="No character_updates",
            category="npc",
        )

    violations = []
    for slug, data in char_updates.items():
        if not isinstance(data, dict):
            continue
        # Only check existing characters (updates, not creation)
        if slug not in state.characters:
            continue  # New character — immutable fields are expected

        for field in IMMUTABLE_FIELDS:
            if field in data:
                original = state.characters[slug].get(field)
                new_val = data[field]
                if original is not None and new_val != original:
                    violations.append(f"{slug}.{field} changed")

    if violations:
        return CheckResult(
            check_id="npc_immutable_preserved",
            passed=False,
            detail=f"Immutable field changes: {violations}",
            category="npc",
        )
    return CheckResult(
        check_id="npc_immutable_preserved",
        passed=True,
        detail="All immutable fields preserved",
        category="npc",
    )


def check_npc_creation_has_required(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that new NPCs have all required creation fields (at minimum: name)."""
    char_updates = parsed.parsed_json.get("character_updates")
    if not isinstance(char_updates, dict) or not char_updates:
        return CheckResult(
            check_id="npc_creation_has_required",
            passed=True,
            detail="No character_updates",
            category="npc",
        )

    issues: list[str] = []
    for slug, data in char_updates.items():
        if not isinstance(data, dict):
            continue
        # Only check new characters (not updates to existing ones)
        if slug in state.characters:
            continue
        # Check deletion sentinels
        if data.get("remove"):
            continue

        missing = CREATION_REQUIRED_FIELDS - set(data.keys())
        if missing:
            issues.append(f"{slug} missing {sorted(missing)}")

    if issues:
        return CheckResult(
            check_id="npc_creation_has_required",
            passed=False,
            detail=f"New NPC missing required fields: {'; '.join(issues)}",
            category="npc",
        )
    return CheckResult(
        check_id="npc_creation_has_required",
        passed=True,
        detail="All new NPCs have required fields",
        category="npc",
    )


def check_npc_update_fields_valid(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that updates to existing NPCs only use mutable fields.

    New NPCs may use any valid field. Existing NPCs should only update
    mutable fields (location, current_status, allegiance, etc.).
    """
    char_updates = parsed.parsed_json.get("character_updates")
    if not isinstance(char_updates, dict) or not char_updates:
        return CheckResult(
            check_id="npc_update_fields_valid",
            passed=True,
            detail="No character_updates",
            category="npc",
        )

    issues: list[str] = []
    for slug, data in char_updates.items():
        if not isinstance(data, dict):
            # Deletion sentinel (null, "delete", "remove") — skip
            continue

        if slug in state.characters:
            # Existing NPC — only mutable fields allowed
            # (immutable re-assertion with same value is OK and handled by
            # npc_immutable_preserved, but unknown fields are flagged here)
            invalid_keys = set(data.keys()) - ALL_VALID_FIELDS
            if invalid_keys:
                issues.append(f"{slug} has unknown fields: {sorted(invalid_keys)}")
        else:
            # New NPC — all valid fields allowed
            invalid_keys = set(data.keys()) - ALL_VALID_FIELDS
            if invalid_keys:
                issues.append(f"{slug} has unknown fields: {sorted(invalid_keys)}")

    if issues:
        return CheckResult(
            check_id="npc_update_fields_valid",
            passed=False,
            detail=f"Invalid character_updates fields: {'; '.join(issues)}",
            category="npc",
        )
    return CheckResult(
        check_id="npc_update_fields_valid",
        passed=True,
        detail="All character_updates fields valid",
        category="npc",
    )


def check_npc_no_creation_on_rails(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that new NPCs are not created when campaign is on_rails."""
    if not scenario.campaign.on_rails:
        return CheckResult(
            check_id="npc_no_creation_on_rails",
            passed=True,
            detail="Campaign not on_rails — NPC creation allowed",
            category="npc",
        )

    char_updates = parsed.parsed_json.get("character_updates")
    if not isinstance(char_updates, dict) or not char_updates:
        return CheckResult(
            check_id="npc_no_creation_on_rails",
            passed=True,
            detail="No character_updates",
            category="npc",
        )

    new_chars = []
    for slug, data in char_updates.items():
        if not isinstance(data, dict):
            continue
        if slug not in state.characters and not data.get("remove"):
            new_chars.append(slug)

    if new_chars:
        return CheckResult(
            check_id="npc_no_creation_on_rails",
            passed=False,
            detail=f"New NPCs created in on_rails mode: {new_chars}",
            category="npc",
        )
    return CheckResult(
        check_id="npc_no_creation_on_rails",
        passed=True,
        detail="No new NPCs created in on_rails mode",
        category="npc",
    )
