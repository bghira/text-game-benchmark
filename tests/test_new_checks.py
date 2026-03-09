"""Tests for new check modules: calendar, give_item, expanded npc checks."""

import json
import pytest

from tgb.checks.calendar import (
    check_calendar_update_valid,
    check_calendar_no_legacy_fields,
)
from tgb.checks.give_item import (
    check_give_item_valid,
    check_give_item_no_double_remove,
)
from tgb.checks.npc import (
    check_npc_slug_valid,
    check_npc_immutable_preserved,
    check_npc_creation_has_required,
    check_npc_update_fields_valid,
    check_npc_no_creation_on_rails,
    MUTABLE_FIELDS,
    IMMUTABLE_FIELDS,
    ALL_VALID_FIELDS,
)
from tgb.checks.json_structure import EXPECTED_TYPES
from tgb.response_parser import ParsedResponse
from tgb.config import Scenario, CampaignSetup, PlayerSetup, TurnSpec
from tgb.prompt_builder import AccumulatedState


def _make_scenario(**kwargs) -> Scenario:
    return Scenario(
        name=kwargs.get("name", "test"),
        description="test",
        tags=[],
        tier="basic",
        campaign=kwargs.get("campaign", CampaignSetup(name="test")),
        player=kwargs.get("player", PlayerSetup()),
        turns=[TurnSpec(action="test")],
    )


def _make_state(scenario=None, **overrides) -> AccumulatedState:
    scenario = scenario or _make_scenario()
    state = AccumulatedState(scenario)
    for k, v in overrides.items():
        setattr(state, k, v)
    return state


def _make_parsed(json_data=None, raw="", parse_error="", is_tool_call=False) -> ParsedResponse:
    if json_data is None:
        json_data = {}
    if not raw and json_data:
        raw = json.dumps(json_data)
    return ParsedResponse(
        raw=raw,
        parsed_json=json_data,
        parse_error=parse_error,
        is_tool_call=is_tool_call,
    )


TURN = TurnSpec(action="test")
P = {}  # empty params


# ── EXPECTED_TYPES completeness ────────────────────────────────────

class TestExpectedTypes:
    def test_has_turn_visibility(self):
        assert "turn_visibility" in EXPECTED_TYPES
        assert EXPECTED_TYPES["turn_visibility"] is dict

    def test_has_calendar_update(self):
        assert "calendar_update" in EXPECTED_TYPES
        assert EXPECTED_TYPES["calendar_update"] is dict

    def test_has_dice_check(self):
        assert "dice_check" in EXPECTED_TYPES
        assert EXPECTED_TYPES["dice_check"] is dict

    def test_has_puzzle_trigger(self):
        assert "puzzle_trigger" in EXPECTED_TYPES
        assert EXPECTED_TYPES["puzzle_trigger"] is dict

    def test_has_minigame_challenge(self):
        assert "minigame_challenge" in EXPECTED_TYPES
        assert EXPECTED_TYPES["minigame_challenge"] is dict

    def test_has_timer_interrupt_fields(self):
        assert "set_timer_interrupt_action" in EXPECTED_TYPES
        assert "set_timer_interrupt_scope" in EXPECTED_TYPES


# ── Calendar checks ───────────────────────────────────────────────

class TestCalendarUpdateValid:
    def test_no_calendar(self):
        parsed = _make_parsed({"narration": "hello"})
        result = check_calendar_update_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_valid_add(self):
        parsed = _make_parsed({"calendar_update": {
            "add": [{"name": "festival", "time_remaining": 3, "time_unit": "days"}],
        }})
        result = check_calendar_update_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_valid_remove(self):
        parsed = _make_parsed({"calendar_update": {
            "remove": ["old-event"],
        }})
        result = check_calendar_update_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_valid_add_and_remove(self):
        parsed = _make_parsed({"calendar_update": {
            "add": [{"name": "new-event", "time_remaining": 1, "time_unit": "hours"}],
            "remove": ["old-event"],
        }})
        result = check_calendar_update_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_not_dict(self):
        parsed = _make_parsed({"calendar_update": "bad"})
        result = check_calendar_update_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed

    def test_unknown_top_key(self):
        parsed = _make_parsed({"calendar_update": {"add": [], "bogus": True}})
        result = check_calendar_update_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed

    def test_add_missing_required(self):
        parsed = _make_parsed({"calendar_update": {
            "add": [{"name": "test"}],  # missing time_remaining and time_unit
        }})
        result = check_calendar_update_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed

    def test_add_bad_time_unit(self):
        parsed = _make_parsed({"calendar_update": {
            "add": [{"name": "x", "time_remaining": 1, "time_unit": "weeks"}],
        }})
        result = check_calendar_update_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed

    def test_add_zero_time_remaining(self):
        parsed = _make_parsed({"calendar_update": {
            "add": [{"name": "x", "time_remaining": 0, "time_unit": "hours"}],
        }})
        result = check_calendar_update_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed

    def test_add_name_too_long(self):
        parsed = _make_parsed({"calendar_update": {
            "add": [{"name": "x" * 200, "time_remaining": 1, "time_unit": "hours"}],
        }})
        result = check_calendar_update_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed

    def test_remove_non_string(self):
        parsed = _make_parsed({"calendar_update": {
            "remove": [123],
        }})
        result = check_calendar_update_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed

    def test_add_with_known_by(self):
        parsed = _make_parsed({"calendar_update": {
            "add": [{
                "name": "secret-meeting",
                "time_remaining": 2,
                "time_unit": "hours",
                "known_by": ["alice", "bob"],
            }],
        }})
        result = check_calendar_update_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_add_known_by_not_list(self):
        parsed = _make_parsed({"calendar_update": {
            "add": [{
                "name": "x",
                "time_remaining": 2,
                "time_unit": "hours",
                "known_by": "alice",
            }],
        }})
        result = check_calendar_update_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed


class TestCalendarNoLegacyFields:
    def test_no_legacy(self):
        parsed = _make_parsed({"calendar_update": {
            "add": [{"name": "x", "time_remaining": 1, "time_unit": "hours"}],
        }})
        result = check_calendar_no_legacy_fields(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_hours_remaining_legacy(self):
        parsed = _make_parsed({"calendar_update": {
            "add": [{"name": "x", "hours_remaining": 5}],
        }})
        result = check_calendar_no_legacy_fields(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed

    def test_days_remaining_legacy(self):
        parsed = _make_parsed({"calendar_update": {
            "add": [{"name": "x", "days_remaining": 3}],
        }})
        result = check_calendar_no_legacy_fields(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed

    def test_status_legacy(self):
        parsed = _make_parsed({"calendar_update": {
            "add": [{"name": "x", "time_remaining": 1, "time_unit": "hours", "status": "pending"}],
        }})
        result = check_calendar_no_legacy_fields(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed

    def test_no_calendar(self):
        parsed = _make_parsed({"narration": "hi"})
        result = check_calendar_no_legacy_fields(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed


# ── Give-item checks ──────────────────────────────────────────────

class TestGiveItemValid:
    def test_no_give_item(self):
        parsed = _make_parsed({"narration": "hello"})
        result = check_give_item_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_valid_with_actor_id(self):
        parsed = _make_parsed({"give_item": {
            "item": "golden key",
            "to_actor_id": "123456",
        }})
        result = check_give_item_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_valid_with_mention(self):
        parsed = _make_parsed({"give_item": {
            "item": "sword",
            "to_discord_mention": "<@123456>",
        }})
        result = check_give_item_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_valid_with_nick_mention(self):
        parsed = _make_parsed({"give_item": {
            "item": "potion",
            "to_discord_mention": "<@!789>",
        }})
        result = check_give_item_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_missing_item(self):
        parsed = _make_parsed({"give_item": {
            "to_actor_id": "123",
        }})
        result = check_give_item_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed

    def test_empty_item(self):
        parsed = _make_parsed({"give_item": {
            "item": "",
            "to_actor_id": "123",
        }})
        result = check_give_item_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed

    def test_no_recipient(self):
        parsed = _make_parsed({"give_item": {"item": "key"}})
        result = check_give_item_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed

    def test_bad_mention_format(self):
        parsed = _make_parsed({"give_item": {
            "item": "key",
            "to_discord_mention": "@bob",
        }})
        result = check_give_item_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed

    def test_not_dict(self):
        parsed = _make_parsed({"give_item": "key"})
        result = check_give_item_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed


class TestGiveItemNoDoubleRemove:
    def test_no_give_item(self):
        parsed = _make_parsed({"narration": "hello"})
        result = check_give_item_no_double_remove(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_no_conflict(self):
        parsed = _make_parsed({
            "give_item": {"item": "sword", "to_actor_id": "123"},
            "player_state_update": {"inventory_remove": ["shield"]},
        })
        result = check_give_item_no_double_remove(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_double_remove(self):
        parsed = _make_parsed({
            "give_item": {"item": "sword", "to_actor_id": "123"},
            "player_state_update": {"inventory_remove": ["sword"]},
        })
        result = check_give_item_no_double_remove(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed

    def test_case_insensitive_double_remove(self):
        parsed = _make_parsed({
            "give_item": {"item": "Golden Key", "to_actor_id": "123"},
            "player_state_update": {"inventory_remove": ["golden key"]},
        })
        result = check_give_item_no_double_remove(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed

    def test_no_inventory_remove(self):
        parsed = _make_parsed({
            "give_item": {"item": "sword", "to_actor_id": "123"},
            "player_state_update": {"hp": 10},
        })
        result = check_give_item_no_double_remove(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed


# ── NPC creation checks ──────────────────────────────────────────

class TestNpcCreationHasRequired:
    def test_no_updates(self):
        parsed = _make_parsed({"narration": "hello"})
        result = check_npc_creation_has_required(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_new_npc_with_name(self):
        parsed = _make_parsed({"character_updates": {
            "bob-the-guard": {"name": "Bob the Guard", "personality": "stern"},
        }})
        result = check_npc_creation_has_required(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_new_npc_missing_name(self):
        parsed = _make_parsed({"character_updates": {
            "bob-the-guard": {"personality": "stern"},
        }})
        result = check_npc_creation_has_required(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed

    def test_existing_npc_no_name_required(self):
        state = _make_state()
        state.characters["bob-the-guard"] = {"name": "Bob the Guard"}
        parsed = _make_parsed({"character_updates": {
            "bob-the-guard": {"current_status": "sleeping"},
        }})
        result = check_npc_creation_has_required(parsed, _make_scenario(), TURN, state, P)
        assert result.passed

    def test_deletion_no_check(self):
        parsed = _make_parsed({"character_updates": {
            "bob-the-guard": None,
        }})
        result = check_npc_creation_has_required(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed


class TestNpcUpdateFieldsValid:
    def test_no_updates(self):
        parsed = _make_parsed({"narration": "hello"})
        result = check_npc_update_fields_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_valid_mutable_fields(self):
        state = _make_state()
        state.characters["guard"] = {"name": "Guard", "location": "gate"}
        parsed = _make_parsed({"character_updates": {
            "guard": {"location": "dungeon", "current_status": "angry"},
        }})
        result = check_npc_update_fields_valid(parsed, _make_scenario(), TURN, state, P)
        assert result.passed

    def test_unknown_field_on_existing(self):
        state = _make_state()
        state.characters["guard"] = {"name": "Guard"}
        parsed = _make_parsed({"character_updates": {
            "guard": {"secret_power": "flight"},
        }})
        result = check_npc_update_fields_valid(parsed, _make_scenario(), TURN, state, P)
        assert not result.passed

    def test_unknown_field_on_new(self):
        parsed = _make_parsed({"character_updates": {
            "wizard": {"name": "Gandalf", "magic_level": 99},
        }})
        result = check_npc_update_fields_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed

    def test_all_valid_fields_accepted(self):
        """All fields from IMMUTABLE_FIELDS and MUTABLE_FIELDS should be accepted."""
        data = {field: "test" for field in ALL_VALID_FIELDS}
        parsed = _make_parsed({"character_updates": {"new-npc": data}})
        result = check_npc_update_fields_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed


class TestNpcNoCreationOnRails:
    def test_not_on_rails(self):
        parsed = _make_parsed({"character_updates": {
            "new-guy": {"name": "New Guy"},
        }})
        result = check_npc_no_creation_on_rails(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_on_rails_no_creation(self):
        scenario = _make_scenario(campaign=CampaignSetup(name="test", on_rails=True))
        state = _make_state(scenario=scenario)
        state.characters["guard"] = {"name": "Guard"}
        parsed = _make_parsed({"character_updates": {
            "guard": {"current_status": "sleeping"},
        }})
        result = check_npc_no_creation_on_rails(parsed, scenario, TURN, state, P)
        assert result.passed

    def test_on_rails_blocks_creation(self):
        scenario = _make_scenario(campaign=CampaignSetup(name="test", on_rails=True))
        state = _make_state(scenario=scenario)
        parsed = _make_parsed({"character_updates": {
            "new-guy": {"name": "New Guy"},
        }})
        result = check_npc_no_creation_on_rails(parsed, scenario, TURN, state, P)
        assert not result.passed

    def test_on_rails_allows_deletion(self):
        scenario = _make_scenario(campaign=CampaignSetup(name="test", on_rails=True))
        state = _make_state(scenario=scenario)
        parsed = _make_parsed({"character_updates": {
            "old-npc": None,
        }})
        result = check_npc_no_creation_on_rails(parsed, scenario, TURN, state, P)
        assert result.passed


# ── NPC field constants ───────────────────────────────────────────

class TestNpcFieldConstants:
    def test_immutable_mutable_disjoint(self):
        assert IMMUTABLE_FIELDS.isdisjoint(MUTABLE_FIELDS)

    def test_all_valid_is_union(self):
        assert ALL_VALID_FIELDS == IMMUTABLE_FIELDS | MUTABLE_FIELDS

    def test_key_mutable_fields_present(self):
        assert "location" in MUTABLE_FIELDS
        assert "current_status" in MUTABLE_FIELDS
        assert "allegiance" in MUTABLE_FIELDS
        assert "deceased_reason" in MUTABLE_FIELDS

    def test_key_immutable_fields_present(self):
        assert "name" in IMMUTABLE_FIELDS
        assert "personality" in IMMUTABLE_FIELDS
        assert "background" in IMMUTABLE_FIELDS
        assert "appearance" in IMMUTABLE_FIELDS
        assert "speech_style" in IMMUTABLE_FIELDS
