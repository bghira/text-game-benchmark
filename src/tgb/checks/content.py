"""Content checks: scene_image_prompt_present, rulebook_adherent."""

from __future__ import annotations

from typing import Any

from tgb.checks.base import CheckResult
from tgb.config import Scenario, TurnSpec
from tgb.prompt_builder import AccumulatedState
from tgb.response_parser import ParsedResponse


def check_scene_image_prompt_present(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that scene_image_prompt is non-empty when visual changes occur.

    Params:
        expect_visual_change: bool — if True, scene_image_prompt is required
    """
    expect_change = params.get("expect_visual_change", True)
    data = parsed.parsed_json
    prompt_val = data.get("scene_image_prompt", "")

    if not expect_change:
        return CheckResult(
            check_id="scene_image_prompt_present",
            passed=True,
            detail="No visual change expected",
            category="content",
        )

    # Also check if location changed (auto-detect visual change)
    player_update = data.get("player_state_update", {})
    location_changed = isinstance(player_update, dict) and "location" in player_update

    if location_changed or expect_change:
        if not prompt_val or not isinstance(prompt_val, str) or not prompt_val.strip():
            return CheckResult(
                check_id="scene_image_prompt_present",
                passed=False,
                detail="Visual change occurred but scene_image_prompt is missing/empty",
                category="content",
            )

    return CheckResult(
        check_id="scene_image_prompt_present",
        passed=True,
        detail=f"scene_image_prompt present ({len(str(prompt_val))} chars)",
        category="content",
    )


def check_rulebook_adherent(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that narration doesn't violate listed constraints.

    Params:
        constraints: list[str] — constraint descriptions to check against
    """
    narration = parsed.parsed_json.get("narration", "")
    if not isinstance(narration, str):
        return CheckResult(
            check_id="rulebook_adherent",
            passed=True,
            detail="No narration to check",
            category="content",
        )

    # Get constraints from params or source material
    constraints = params.get("constraints", [])
    if not constraints and scenario.source_material:
        constraints = scenario.source_material.constraints

    if not constraints:
        return CheckResult(
            check_id="rulebook_adherent",
            passed=True,
            detail="No constraints defined",
            category="content",
        )

    # Basic keyword-based constraint checking
    # For more nuanced checking, use judge:rulebook_adherent
    violations = []
    narration_lower = narration.lower()

    for constraint in constraints:
        # Extract forbidden terms from constraint (simple heuristic)
        if constraint.lower().startswith("no "):
            forbidden = constraint[3:].strip().lower()
            if forbidden in narration_lower:
                violations.append(constraint)
        elif constraint.lower().startswith("never "):
            forbidden = constraint[6:].strip().lower()
            if forbidden in narration_lower:
                violations.append(constraint)

    if violations:
        return CheckResult(
            check_id="rulebook_adherent",
            passed=False,
            detail=f"Constraint violations: {violations[:3]}",
            category="content",
        )
    return CheckResult(
        check_id="rulebook_adherent",
        passed=True,
        detail=f"Passed {len(constraints)} constraint checks",
        category="content",
    )
