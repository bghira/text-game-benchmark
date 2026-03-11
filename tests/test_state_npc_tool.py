"""Tests for game_time_period, state_update_required_fields, summary_update, party_status, npc_relationships, tool_format updates."""

import json

from tgb.checks.state_mgmt import (
    check_game_time_period_valid,
    check_state_update_required_fields,
    check_summary_update_valid,
    check_party_status_valid,
)
from tgb.checks.npc import check_npc_relationships_valid
from tgb.checks.tool_checks import check_tool_format_valid
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
P = {}


# ── game_time_period_valid ──────────────────────────────────────

class TestGameTimePeriodValid:
    def test_valid_morning(self):
        parsed = _make_parsed({
            "state_update": {"game_time": {"day": 1, "hour": 8, "period": "morning"}},
        })
        result = check_game_time_period_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_valid_night(self):
        parsed = _make_parsed({
            "state_update": {"game_time": {"day": 1, "hour": 22, "period": "night"}},
        })
        result = check_game_time_period_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_invalid_period(self):
        parsed = _make_parsed({
            "state_update": {"game_time": {"day": 1, "hour": 12, "period": "noon"}},
        })
        result = check_game_time_period_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed
        assert "noon" in result.detail

    def test_no_period_ok(self):
        parsed = _make_parsed({
            "state_update": {"game_time": {"day": 1, "hour": 12}},
        })
        result = check_game_time_period_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_no_state_update(self):
        parsed = _make_parsed({"narration": "x"})
        result = check_game_time_period_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_tool_call_skipped(self):
        parsed = _make_parsed({"tool_call": "memory_search"}, is_tool_call=True)
        result = check_game_time_period_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_all_valid_periods(self):
        for period in ("morning", "afternoon", "evening", "night"):
            parsed = _make_parsed({
                "state_update": {"game_time": {"day": 1, "hour": 12, "period": period}},
            })
            result = check_game_time_period_valid(parsed, _make_scenario(), TURN, _make_state(), P)
            assert result.passed, f"period '{period}' should be valid"


# ── state_update_required_fields ─────────────────────────────────

class TestStateUpdateRequiredFields:
    def test_all_present(self):
        parsed = _make_parsed({
            "state_update": {
                "game_time": {"day": 1, "hour": 10},
                "current_chapter": 0,
                "current_scene": 1,
            },
        })
        result = check_state_update_required_fields(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_missing_current_chapter(self):
        parsed = _make_parsed({
            "state_update": {
                "game_time": {"day": 1, "hour": 10},
                "current_scene": 1,
            },
        })
        result = check_state_update_required_fields(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed
        assert "current_chapter" in result.detail

    def test_missing_all(self):
        parsed = _make_parsed({
            "state_update": {"mood": "tense"},
        })
        result = check_state_update_required_fields(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed

    def test_no_state_update(self):
        parsed = _make_parsed({"narration": "x"})
        result = check_state_update_required_fields(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed

    def test_tool_call_skipped(self):
        parsed = _make_parsed({"tool_call": "recent_turns"}, is_tool_call=True)
        result = check_state_update_required_fields(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_custom_required_via_params(self):
        parsed = _make_parsed({
            "state_update": {"game_time": {"day": 1}},
        })
        result = check_state_update_required_fields(
            parsed, _make_scenario(), TURN, _make_state(),
            {"required": ["game_time"]},
        )
        assert result.passed


# ── summary_update_valid ────────────────────────────────────────

class TestSummaryUpdateValid:
    def test_valid_summary(self):
        parsed = _make_parsed({
            "narration": "x",
            "summary_update": "The guard revealed the secret passage behind the bookshelf.",
        })
        result = check_summary_update_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_no_summary_ok(self):
        parsed = _make_parsed({"narration": "x"})
        result = check_summary_update_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_empty_summary(self):
        parsed = _make_parsed({"narration": "x", "summary_update": ""})
        result = check_summary_update_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed

    def test_summary_not_string(self):
        parsed = _make_parsed({"narration": "x", "summary_update": 42})
        result = check_summary_update_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed

    def test_summary_too_long(self):
        parsed = _make_parsed({
            "narration": "x",
            "summary_update": "A" * 600,
        })
        result = check_summary_update_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed
        assert "600" in result.detail

    def test_tool_call_skipped(self):
        parsed = _make_parsed({"tool_call": "recent_turns"}, is_tool_call=True)
        result = check_summary_update_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed


# ── party_status_valid ──────────────────────────────────────────

class TestPartyStatusValid:
    def test_main_party(self):
        parsed = _make_parsed({
            "player_state_update": {"party_status": "main_party"},
        })
        result = check_party_status_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_new_path(self):
        parsed = _make_parsed({
            "player_state_update": {"party_status": "new_path"},
        })
        result = check_party_status_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_invalid_status(self):
        parsed = _make_parsed({
            "player_state_update": {"party_status": "solo"},
        })
        result = check_party_status_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed
        assert "solo" in result.detail

    def test_no_party_status_ok(self):
        parsed = _make_parsed({
            "player_state_update": {"hp": 50},
        })
        result = check_party_status_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_no_player_update_ok(self):
        parsed = _make_parsed({"narration": "x"})
        result = check_party_status_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed


# ── npc_relationships_valid ─────────────────────────────────────

class TestNpcRelationshipsValid:
    def test_no_updates(self):
        parsed = _make_parsed({"narration": "x"})
        result = check_npc_relationships_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_valid_relationships(self):
        parsed = _make_parsed({"character_updates": {
            "guard": {
                "relationships": {
                    "captain": {"status": "loyal", "knows_about": "patrol routes"},
                    "player": {"status": "suspicious", "dynamic": "warming up"},
                },
            },
        }})
        result = check_npc_relationships_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_relationships_not_dict(self):
        parsed = _make_parsed({"character_updates": {
            "guard": {"relationships": "friends with captain"},
        }})
        result = check_npc_relationships_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed
        assert "guard.relationships" in result.detail

    def test_relationship_entry_not_dict(self):
        parsed = _make_parsed({"character_updates": {
            "guard": {
                "relationships": {
                    "captain": "loyal",
                },
            },
        }})
        result = check_npc_relationships_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed
        assert "captain" in result.detail

    def test_no_relationships_field_ok(self):
        parsed = _make_parsed({"character_updates": {
            "guard": {"current_status": "patrolling"},
        }})
        result = check_npc_relationships_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed


# ── tool_format_valid updates ───────────────────────────────────

class TestToolFormatValidUpdated:
    def test_communication_rules_valid(self):
        parsed = _make_parsed(
            {"tool_call": "communication_rules", "keys": ["GM-RULE-COMMUNICATION-SOFTENING"]},
            is_tool_call=True,
        )
        result = check_tool_format_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_communication_rules_missing_keys(self):
        parsed = _make_parsed(
            {"tool_call": "communication_rules"},
            is_tool_call=True,
        )
        result = check_tool_format_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed
        assert "keys" in result.detail

    def test_story_outline_valid(self):
        parsed = _make_parsed(
            {"tool_call": "story_outline", "chapter": "chapter-1"},
            is_tool_call=True,
        )
        result = check_tool_format_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_story_outline_missing_chapter(self):
        parsed = _make_parsed(
            {"tool_call": "story_outline"},
            is_tool_call=True,
        )
        result = check_tool_format_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed
        assert "chapter" in result.detail

    def test_name_generate_with_optional(self):
        parsed = _make_parsed(
            {"tool_call": "name_generate", "origins": ["english"], "gender": "f", "count": 3},
            is_tool_call=True,
        )
        result = check_tool_format_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_source_browse_with_optional(self):
        parsed = _make_parsed(
            {"tool_call": "source_browse", "document_key": "map-data", "wildcard": "*.txt"},
            is_tool_call=True,
        )
        result = check_tool_format_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_recent_turns_with_limit(self):
        parsed = _make_parsed(
            {"tool_call": "recent_turns", "player_slugs": ["player-1"], "npc_slugs": ["guard"], "limit": 5},
            is_tool_call=True,
        )
        result = check_tool_format_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed
