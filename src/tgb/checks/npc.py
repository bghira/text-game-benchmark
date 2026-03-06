"""NPC checks: npc_slug_valid, npc_immutable_preserved."""

from __future__ import annotations

import re
from typing import Any

from tgb.checks.base import CheckResult
from tgb.config import Scenario, TurnSpec
from tgb.prompt_builder import AccumulatedState
from tgb.response_parser import ParsedResponse

SLUG_PATTERN = re.compile(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$")

IMMUTABLE_FIELDS = {"name", "personality", "background", "appearance", "speech_style"}


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
