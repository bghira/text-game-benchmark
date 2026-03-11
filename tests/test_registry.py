"""Tests for check registry completeness."""

from tgb.checks.registry import CHECKS, get_check, list_checks


class TestRegistryCompleteness:
    """Ensure all check modules are registered."""

    def test_calendar_checks_registered(self):
        assert "calendar_update_valid" in CHECKS
        assert "calendar_no_legacy_fields" in CHECKS

    def test_give_item_checks_registered(self):
        assert "give_item_valid" in CHECKS
        assert "give_item_no_double_remove" in CHECKS

    def test_npc_new_checks_registered(self):
        assert "npc_creation_has_required" in CHECKS
        assert "npc_update_fields_valid" in CHECKS
        assert "npc_no_creation_on_rails" in CHECKS

    def test_writing_craft_checks_registered(self):
        assert "narration_no_echo" in CHECKS
        assert "narration_no_therapist_speak" in CHECKS
        assert "narration_not_abstract" in CHECKS

    def test_existing_checks_still_registered(self):
        existing = [
            "json_valid", "json_keys_present", "json_types_correct",
            "xp_range", "reasoning_present", "reasoning_concise",
            "narration_length", "narration_no_recap",
            "npc_slug_valid", "npc_immutable_preserved",
            "timer_fields_valid", "sms_tool_used",
            "visibility_fields_valid", "visibility_scope_present",
        ]
        for check_id in existing:
            assert check_id in CHECKS, f"{check_id} missing from registry"

    def test_get_check_returns_callable(self):
        fn = get_check("calendar_update_valid")
        assert callable(fn)

    def test_get_check_unknown_raises(self):
        import pytest
        with pytest.raises(KeyError):
            get_check("nonexistent_check_xyz")

    def test_list_checks_sorted(self):
        checks = list_checks()
        assert checks == sorted(checks)

    def test_total_check_count(self):
        """Sanity check: we should have a reasonable number of checks."""
        assert len(CHECKS) >= 40
