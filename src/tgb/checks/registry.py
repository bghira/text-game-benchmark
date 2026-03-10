"""Check lookup by ID."""

from __future__ import annotations

from typing import Any, Callable

from tgb.checks.json_structure import (
    check_json_valid,
    check_json_keys_present,
    check_json_types_correct,
    check_xp_range,
    check_reasoning_present,
)
from tgb.checks.narrative import (
    check_reasoning_concise,
    check_narration_length,
    check_narration_no_recap,
    check_no_inventory_in_narration,
    check_no_markdown_in_response,
)
from tgb.checks.state_mgmt import (
    check_state_nested,
    check_state_null_prune,
)
from tgb.checks.location import check_location_coherent
from tgb.checks.tool_checks import check_tool_called, check_tool_format_valid
from tgb.checks.npc import (
    check_npc_slug_valid,
    check_npc_immutable_preserved,
    check_npc_creation_has_required,
    check_npc_update_fields_valid,
    check_npc_no_creation_on_rails,
)
from tgb.checks.agency import check_consent_respected, check_player_agency_respected
from tgb.checks.content import (
    check_scene_image_prompt_present,
    check_rulebook_adherent,
)
from tgb.checks.timer import (
    check_timer_fields_valid,
    check_timer_no_countdown_in_narration,
    check_timer_grounded,
    check_timer_delay_appropriate,
    check_no_gratuitous_timer,
)
from tgb.checks.sms import (
    check_sms_tool_used,
    check_sms_write_fields_valid,
    check_sms_both_sides_recorded,
    check_sms_no_context_leak,
    check_sms_thread_slug_stable,
    check_no_sms_in_wrong_era,
)
from tgb.checks.subplot import (
    plot_thread_fields_valid,
    plot_thread_target_reasonable,
    plot_thread_not_orphaned,
    chapter_fields_valid,
    chapter_scene_progression,
    consequence_fields_valid,
    consequence_severity_proportional,
)
from tgb.checks.calendar import (
    check_calendar_update_valid,
    check_calendar_no_legacy_fields,
)
from tgb.checks.give_item import (
    check_give_item_valid,
    check_give_item_no_double_remove,
)
from tgb.checks.privacy import (
    check_visibility_fields_valid,
    check_visibility_scope_present,
    check_visibility_default_respected,
    check_visibility_player_slugs_known,
    check_visibility_npc_slugs_known,
    check_visibility_no_narration_leak,
    check_calendar_known_by_valid,
    check_calendar_target_player_valid,
    check_no_public_leak_in_private_turn,
    check_sms_not_in_narration,
    check_sms_turn_private,
)

from tgb.checks.base import CheckFn

# Master registry: check_id -> check function
CHECKS: dict[str, CheckFn] = {
    "json_valid": check_json_valid,
    "json_keys_present": check_json_keys_present,
    "json_types_correct": check_json_types_correct,
    "xp_range": check_xp_range,
    "reasoning_present": check_reasoning_present,
    "reasoning_concise": check_reasoning_concise,
    "narration_length": check_narration_length,
    "narration_no_recap": check_narration_no_recap,
    "no_inventory_in_narration": check_no_inventory_in_narration,
    "no_markdown_in_response": check_no_markdown_in_response,
    "state_nested": check_state_nested,
    "state_null_prune": check_state_null_prune,
    "location_coherent": check_location_coherent,
    "tool_called": check_tool_called,
    "tool_format_valid": check_tool_format_valid,
    "npc_slug_valid": check_npc_slug_valid,
    "npc_immutable_preserved": check_npc_immutable_preserved,
    "npc_creation_has_required": check_npc_creation_has_required,
    "npc_update_fields_valid": check_npc_update_fields_valid,
    "npc_no_creation_on_rails": check_npc_no_creation_on_rails,
    "consent_respected": check_consent_respected,
    "player_agency_respected": check_player_agency_respected,
    "scene_image_prompt_present": check_scene_image_prompt_present,
    "rulebook_adherent": check_rulebook_adherent,
    # Timer checks
    "timer_fields_valid": check_timer_fields_valid,
    "timer_no_countdown_in_narration": check_timer_no_countdown_in_narration,
    "timer_grounded": check_timer_grounded,
    "timer_delay_appropriate": check_timer_delay_appropriate,
    "no_gratuitous_timer": check_no_gratuitous_timer,
    # SMS checks
    "sms_tool_used": check_sms_tool_used,
    "sms_write_fields_valid": check_sms_write_fields_valid,
    "sms_both_sides_recorded": check_sms_both_sides_recorded,
    "sms_no_context_leak": check_sms_no_context_leak,
    "sms_thread_slug_stable": check_sms_thread_slug_stable,
    "no_sms_in_wrong_era": check_no_sms_in_wrong_era,
    # Subplot checks
    "plot_thread_fields_valid": plot_thread_fields_valid,
    "plot_thread_target_reasonable": plot_thread_target_reasonable,
    "plot_thread_not_orphaned": plot_thread_not_orphaned,
    "chapter_fields_valid": chapter_fields_valid,
    "chapter_scene_progression": chapter_scene_progression,
    "consequence_fields_valid": consequence_fields_valid,
    "consequence_severity_proportional": consequence_severity_proportional,
    # Calendar checks
    "calendar_update_valid": check_calendar_update_valid,
    "calendar_no_legacy_fields": check_calendar_no_legacy_fields,
    # Give-item checks
    "give_item_valid": check_give_item_valid,
    "give_item_no_double_remove": check_give_item_no_double_remove,
    # Privacy checks
    "visibility_fields_valid": check_visibility_fields_valid,
    "visibility_scope_present": check_visibility_scope_present,
    "visibility_default_respected": check_visibility_default_respected,
    "visibility_player_slugs_known": check_visibility_player_slugs_known,
    "visibility_npc_slugs_known": check_visibility_npc_slugs_known,
    "visibility_no_narration_leak": check_visibility_no_narration_leak,
    "calendar_known_by_valid": check_calendar_known_by_valid,
    "calendar_target_player_valid": check_calendar_target_player_valid,
    "no_public_leak_in_private_turn": check_no_public_leak_in_private_turn,
    "sms_not_in_narration": check_sms_not_in_narration,
    "sms_turn_private": check_sms_turn_private,
}


def get_check(check_id: str) -> CheckFn:
    """Look up a check function by ID. Raises KeyError for unknown IDs."""
    if check_id not in CHECKS:
        raise KeyError(
            f"Unknown check '{check_id}'. "
            f"Available: {sorted(CHECKS.keys())}"
        )
    return CHECKS[check_id]


def list_checks() -> list[str]:
    """Return sorted list of all available check IDs."""
    return sorted(CHECKS.keys())
