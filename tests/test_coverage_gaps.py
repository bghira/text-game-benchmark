"""Tests for coverage gap checks: location_updates, story_progression, tool_calls array,
filing-cabinet narration ban, NPC age/gender, present_characters exclusion,
and round-2 gap fixes (inline allowlist, song_search, game_time ranges, etc.)."""

import json
import pytest

from tgb.checks.location import check_location_updates_valid, PRIORITY_ALIASES, VALID_PRIORITY_VALUES
from tgb.checks.state_mgmt import check_story_progression_valid, check_game_time_range_valid
from tgb.checks.tool_checks import (
    check_inline_tool_calls_valid,
    check_tool_format_valid,
    INLINE_TOOL_CALLS_ALLOWED,
)
from tgb.checks.narrative import check_narration_no_filing_cabinet
from tgb.checks.npc import (
    check_npc_creation_has_required,
    check_npc_immutable_preserved,
    check_npc_gender_format_valid,
    check_npc_update_fields_valid,
    IMMUTABLE_FIELDS,
    CREATION_REQUIRED_FIELDS,
    MUTABLE_FIELDS,
)
from tgb.checks.scene_output import check_beat_narration_no_dialogue, BEAT_TYPE_VALUES
from tgb.checks.multiplayer import check_present_characters_no_players
from tgb.checks.registry import CHECKS
from tgb.response_parser import ParsedResponse
from tgb.config import Scenario, CampaignSetup, PlayerSetup, TurnSpec
from tgb.prompt_builder import AccumulatedState


def _make_scenario(**kwargs) -> Scenario:
    return Scenario(
        name="test",
        description="test",
        tags=[],
        tier="basic",
        campaign=kwargs.get("campaign", CampaignSetup(name="test")),
        player=kwargs.get("player", PlayerSetup()),
        turns=[TurnSpec(action=kwargs.get("action", "test"))],
        party=kwargs.get("party", []),
    )


def _make_mp_scenario(**kwargs) -> Scenario:
    party = kwargs.get("party", [
        {
            "actor_id": "100000020",
            "character_name": "Rico Vega",
            "player_slug": "rico-vega",
            "location": "rooftop",
            "is_actor": True,
        },
        {
            "actor_id": "100000021",
            "character_name": "Mara Chen",
            "player_slug": "mara-chen",
            "location": "rooftop",
            "is_actor": False,
        },
    ])
    return Scenario(
        name="test",
        description="test",
        tags=[],
        tier="basic",
        campaign=kwargs.get("campaign", CampaignSetup(name="test", multi_player=True)),
        player=kwargs.get("player", PlayerSetup(
            state={"character_name": "Rico Vega", "location": "rooftop"},
        )),
        turns=[TurnSpec(action=kwargs.get("action", "test"))],
        party=party,
    )


def _make_state(scenario=None, **overrides) -> AccumulatedState:
    scenario = scenario or _make_scenario()
    state = AccumulatedState(scenario)
    for k, v in overrides.items():
        setattr(state, k, v)
    return state


def _make_parsed(json_data=None, raw="", is_tool_call=False) -> ParsedResponse:
    if json_data is None:
        json_data = {}
    if not raw:
        raw = json.dumps(json_data)
    return ParsedResponse(raw=raw, parsed_json=json_data, is_tool_call=is_tool_call)


# ── location_updates checks ─────────────────────────────────


class TestLocationUpdatesValid:
    def test_skip_when_absent(self):
        result = check_location_updates_valid(
            _make_parsed({"narration": "test"}), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert result.passed
        assert "skipped" in result.detail

    def test_valid_simple(self):
        data = {"location_updates": {
            "hotel-lobby": {"atmosphere": "quiet", "security": "lax"},
        }}
        result = check_location_updates_valid(
            _make_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert result.passed

    def test_valid_priority_wrapped(self):
        data = {"location_updates": {
            "hotel-lobby": {
                "atmosphere": {"value": "tense after the gunshot", "priority": "critical"},
            },
        }}
        result = check_location_updates_valid(
            _make_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert result.passed

    def test_not_a_dict(self):
        data = {"location_updates": "hotel-lobby"}
        result = check_location_updates_valid(
            _make_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert not result.passed
        assert "dict" in result.detail

    def test_bad_slug_format(self):
        data = {"location_updates": {
            "Hotel Lobby": {"atmosphere": "quiet"},
        }}
        result = check_location_updates_valid(
            _make_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert not result.passed
        assert "slug" in result.detail

    def test_location_value_not_dict(self):
        data = {"location_updates": {
            "hotel-lobby": "quiet",
        }}
        result = check_location_updates_valid(
            _make_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert not result.passed

    def test_null_value_for_deletion(self):
        data = {"location_updates": {
            "hotel-lobby": None,
        }}
        result = check_location_updates_valid(
            _make_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert result.passed

    def test_invalid_priority_value(self):
        data = {"location_updates": {
            "hotel-lobby": {
                "atmosphere": {"value": "tense", "priority": "ultra"},
            },
        }}
        result = check_location_updates_valid(
            _make_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert not result.passed
        assert "priority" in result.detail

    def test_priority_wrapped_missing_value_key(self):
        data = {"location_updates": {
            "hotel-lobby": {
                "atmosphere": {"priority": "critical"},
            },
        }}
        result = check_location_updates_valid(
            _make_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert not result.passed
        assert "value" in result.detail


# ── story_progression checks ────────────────────────────────


class TestStoryProgressionValid:
    def test_skip_when_absent(self):
        result = check_story_progression_valid(
            _make_parsed({"narration": "test"}), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert result.passed
        assert "skipped" in result.detail

    def test_valid_progression(self):
        data = {"story_progression": {
            "advance": True,
            "target": "next-scene",
            "reason": "Player completed the puzzle",
        }}
        result = check_story_progression_valid(
            _make_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert result.passed

    def test_not_a_dict(self):
        data = {"story_progression": "next-scene"}
        result = check_story_progression_valid(
            _make_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert not result.passed

    def test_invalid_target(self):
        data = {"story_progression": {
            "advance": True,
            "target": "skip-ahead",
        }}
        result = check_story_progression_valid(
            _make_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert not result.passed
        assert "target" in result.detail

    def test_target_with_underscores_normalized(self):
        data = {"story_progression": {
            "advance": True,
            "target": "next_scene",
        }}
        result = check_story_progression_valid(
            _make_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert result.passed

    def test_advance_not_bool(self):
        data = {"story_progression": {
            "advance": "yes",
            "target": "hold",
        }}
        result = check_story_progression_valid(
            _make_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert not result.passed
        assert "advance" in result.detail

    def test_hold_target_valid(self):
        data = {"story_progression": {
            "advance": False,
            "target": "hold",
            "reason": "Scene not ready to advance",
        }}
        result = check_story_progression_valid(
            _make_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert result.passed


# ── inline tool_calls array checks ──────────────────────────


class TestInlineToolCallsValid:
    def test_skip_when_absent(self):
        result = check_inline_tool_calls_valid(
            _make_parsed({"narration": "test"}), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert result.passed
        assert "skipped" in result.detail

    def test_valid_sms_write(self):
        data = {"tool_calls": [
            {"tool_call": "sms_write", "thread": "saul", "from": "Dale", "to": "Saul", "message": "On my way."},
        ]}
        result = check_inline_tool_calls_valid(
            _make_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert result.passed

    def test_not_a_list(self):
        data = {"tool_calls": {"tool_call": "sms_write"}}
        result = check_inline_tool_calls_valid(
            _make_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert not result.passed
        assert "list" in result.detail

    def test_disallowed_tool(self):
        data = {"tool_calls": [
            {"tool_call": "memory_search", "queries": ["test"]},
        ]}
        result = check_inline_tool_calls_valid(
            _make_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert not result.passed
        assert "not allowed" in result.detail

    def test_entry_not_dict(self):
        data = {"tool_calls": ["sms_write"]}
        result = check_inline_tool_calls_valid(
            _make_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert not result.passed

    def test_missing_tool_call_key(self):
        data = {"tool_calls": [{"thread": "saul"}]}
        result = check_inline_tool_calls_valid(
            _make_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert not result.passed

    def test_valid_sms_schedule(self):
        data = {"tool_calls": [
            {"tool_call": "sms_schedule", "thread": "saul", "from": "Dale", "to": "Saul",
             "message": "Reminder", "delay_seconds": 300},
        ]}
        result = check_inline_tool_calls_valid(
            _make_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert result.passed


# ── filing-cabinet narration ban ────────────────────────────


class TestNarrationNoFilingCabinet:
    def test_skip_no_narration(self):
        result = check_narration_no_filing_cabinet(
            _make_parsed({"tool_call": "memory_search"}), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert result.passed

    def test_clean_narration(self):
        data = {"narration": "Sable looked up sharply, her eyes narrowing at the mention of the Amber Passage."}
        result = check_narration_no_filing_cabinet(
            _make_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert result.passed

    def test_filing_away_detected(self):
        data = {"narration": "She filed that away for later, then turned back to her desk."}
        result = check_narration_no_filing_cabinet(
            _make_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert not result.passed
        assert "Filing-cabinet" in result.detail

    def test_storing_for_later_detected(self):
        data = {"narration": "He stored that away for later and continued walking."}
        result = check_narration_no_filing_cabinet(
            _make_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert not result.passed

    def test_processing_information_detected(self):
        data = {"narration": "She processed the new information carefully before responding."}
        result = check_narration_no_filing_cabinet(
            _make_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert not result.passed


# ── NPC age/gender field updates ────────────────────────────


class TestNpcAgeGenderImmutable:
    def test_age_in_immutable_fields(self):
        assert "age" in IMMUTABLE_FIELDS

    def test_gender_in_immutable_fields(self):
        assert "gender" in IMMUTABLE_FIELDS

    def test_age_in_creation_required(self):
        assert "age" in CREATION_REQUIRED_FIELDS

    def test_gender_in_creation_required(self):
        assert "gender" in CREATION_REQUIRED_FIELDS

    def test_creation_missing_age_gender_fails(self):
        scenario = _make_scenario()
        state = _make_state(scenario)
        data = {"character_updates": {
            "new-npc": {
                "name": "Test",
                "personality": "Bold",
                "background": "Unknown",
                "appearance": "Tall",
                "speech_style": "Short sentences.",
                "location": "lobby",
                "current_status": "standing",
                "allegiance": "neutral",
                "relationship": "stranger",
                # Missing age and gender
            },
        }}
        result = check_npc_creation_has_required(
            _make_parsed(data), scenario, TurnSpec(action="test"), state, {},
        )
        assert not result.passed
        assert "age" in result.detail
        assert "gender" in result.detail

    def test_age_change_on_existing_npc_fails(self):
        scenario = _make_scenario()
        state = _make_state(scenario, characters={
            "sable-marsh": {"name": "Sable Marsh", "age": "55"},
        })
        data = {"character_updates": {
            "sable-marsh": {"age": "30"},
        }}
        result = check_npc_immutable_preserved(
            _make_parsed(data), scenario, TurnSpec(action="test"), state, {},
        )
        assert not result.passed
        assert "age" in result.detail


class TestNpcGenderFormatValid:
    def test_skip_no_character_updates(self):
        result = check_npc_gender_format_valid(
            _make_parsed({"narration": "test"}), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert result.passed

    def test_valid_gender(self):
        data = {"character_updates": {
            "new-npc": {"gender": "cis-female"},
        }}
        result = check_npc_gender_format_valid(
            _make_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert result.passed

    def test_vague_gender_unknown(self):
        data = {"character_updates": {
            "new-npc": {"gender": "unknown"},
        }}
        result = check_npc_gender_format_valid(
            _make_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert not result.passed
        assert "vague" in result.detail

    def test_empty_gender(self):
        data = {"character_updates": {
            "new-npc": {"gender": ""},
        }}
        result = check_npc_gender_format_valid(
            _make_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert not result.passed
        assert "empty" in result.detail

    def test_gender_not_string(self):
        data = {"character_updates": {
            "new-npc": {"gender": 1},
        }}
        result = check_npc_gender_format_valid(
            _make_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert not result.passed

    def test_diegetic_label_passes(self):
        data = {"character_updates": {
            "android-npc": {"gender": "synthetic"},
        }}
        result = check_npc_gender_format_valid(
            _make_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert result.passed


# ── present_characters player exclusion ─────────────────────


class TestPresentCharactersNoPlayers:
    def test_skip_no_character_updates(self):
        result = check_present_characters_no_players(
            _make_parsed({"narration": "test"}), _make_mp_scenario(),
            TurnSpec(action="test"), _make_state(_make_mp_scenario()), {},
        )
        assert result.passed

    def test_skip_no_party(self):
        scenario = _make_scenario()
        result = check_present_characters_no_players(
            _make_parsed({"character_updates": {"some-npc": {"location": "here"}}}),
            scenario, TurnSpec(action="test"), _make_state(scenario), {},
        )
        assert result.passed
        assert "skipped" in result.detail

    def test_npc_in_updates_ok(self):
        scenario = _make_mp_scenario()
        data = {"character_updates": {
            "dispatch": {"current_status": "busy"},
        }}
        result = check_present_characters_no_players(
            _make_parsed(data), scenario,
            TurnSpec(action="test"), _make_state(scenario), {},
        )
        assert result.passed

    def test_player_slug_in_updates_fails(self):
        scenario = _make_mp_scenario()
        data = {"character_updates": {
            "mara-chen": {"location": "fire escape"},
        }}
        result = check_present_characters_no_players(
            _make_parsed(data), scenario,
            TurnSpec(action="test"), _make_state(scenario), {},
        )
        assert not result.passed
        assert "mara-chen" in result.detail

    def test_acting_player_in_updates_fails(self):
        scenario = _make_mp_scenario()
        data = {"character_updates": {
            "rico-vega": {"current_status": "alert"},
        }}
        result = check_present_characters_no_players(
            _make_parsed(data), scenario,
            TurnSpec(action="test"), _make_state(scenario), {},
        )
        assert not result.passed
        assert "rico-vega" in result.detail


# ── Registry coverage ───────────────────────────────────────


class TestRegistryNewChecks:
    @pytest.mark.parametrize("check_id", [
        "location_updates_valid",
        "story_progression_valid",
        "inline_tool_calls_valid",
        "narration_no_filing_cabinet",
        "npc_gender_format_valid",
        "present_characters_no_players",
        "beat_narration_no_dialogue",
        "game_time_range_valid",
    ])
    def test_check_registered(self, check_id):
        assert check_id in CHECKS, f"{check_id} not in registry"


# ── State tracking in prompt_builder ────────────────────────


class TestPromptBuilderStateTracking:
    def test_location_updates_tracked(self):
        scenario = _make_scenario()
        state = AccumulatedState(scenario)
        state.apply({
            "location_updates": {
                "hotel-lobby": {"atmosphere": "tense"},
            },
        })
        facts = state.campaign_state.get("_location_facts", {})
        assert "hotel-lobby" in facts
        assert facts["hotel-lobby"]["atmosphere"] == "tense"

    def test_location_updates_deletion(self):
        scenario = _make_scenario()
        state = AccumulatedState(scenario)
        state.apply({
            "location_updates": {"hotel-lobby": {"atmosphere": "tense"}},
        })
        state.apply({
            "location_updates": {"hotel-lobby": None},
        })
        facts = state.campaign_state.get("_location_facts", {})
        assert "hotel-lobby" not in facts

    def test_story_progression_tracked(self):
        scenario = _make_scenario()
        state = AccumulatedState(scenario)
        state.apply({
            "story_progression": {"advance": True, "target": "next-scene", "reason": "done"},
        })
        prog = state.campaign_state.get("_last_story_progression")
        assert prog is not None
        assert prog["target"] == "next-scene"

    def test_inline_tool_calls_sms_tracked(self):
        scenario = _make_scenario()
        state = AccumulatedState(scenario)
        state.apply({
            "narration": "He texted back.",
            "tool_calls": [
                {"tool_call": "sms_write", "thread": "saul", "from": "Saul",
                 "to": "Dale", "message": "On my way."},
            ],
        })
        threads = state.campaign_state.get("_sms_threads", {})
        assert "saul" in threads
        assert len(threads["saul"]["messages"]) == 1


# ── Round 2: Inline tool_calls expanded allowlist ─────────


class TestInlineToolCallsAllowlist:
    def test_plot_plan_allowed_inline(self):
        data = {"tool_calls": [
            {"tool_call": "plot_plan", "plans": [{"thread": "heist", "status": "active"}]},
        ]}
        result = check_inline_tool_calls_valid(
            _make_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert result.passed

    def test_chapter_plan_allowed_inline(self):
        data = {"tool_calls": [
            {"tool_call": "chapter_plan", "action": "advance_scene", "to_scene": "act-2"},
        ]}
        result = check_inline_tool_calls_valid(
            _make_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert result.passed

    def test_song_search_allowed_inline(self):
        data = {"tool_calls": [
            {"tool_call": "song_search", "query": "feeling good"},
        ]}
        result = check_inline_tool_calls_valid(
            _make_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert result.passed

    def test_allowlist_has_five_entries(self):
        assert INLINE_TOOL_CALLS_ALLOWED == {
            "sms_write", "sms_schedule", "plot_plan", "chapter_plan", "song_search",
        }

    def test_memory_search_still_disallowed(self):
        data = {"tool_calls": [
            {"tool_call": "memory_search", "queries": ["test"]},
        ]}
        result = check_inline_tool_calls_valid(
            _make_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert not result.passed


# ── Round 2: song_search tool validation ──────────────────


class TestSongSearchToolFormat:
    def _make_tool_parsed(self, data):
        return ParsedResponse(raw=json.dumps(data), parsed_json=data, is_tool_call=True)

    def test_song_search_valid(self):
        data = {"tool_call": "song_search", "query": "don't stop believing"}
        result = check_tool_format_valid(
            self._make_tool_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert result.passed

    def test_song_search_with_optional_fields(self):
        data = {"tool_call": "song_search", "query": "hallelujah", "sender": "Marcus", "message": "This one."}
        result = check_tool_format_valid(
            self._make_tool_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert result.passed

    def test_song_search_missing_query(self):
        data = {"tool_call": "song_search", "sender": "Marcus"}
        result = check_tool_format_valid(
            self._make_tool_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert not result.passed
        assert "query" in result.detail

    def test_song_search_extra_key_rejected(self):
        data = {"tool_call": "song_search", "query": "hello", "bogus": True}
        result = check_tool_format_valid(
            self._make_tool_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert not result.passed
        assert "bogus" in result.detail


# ── Round 2: memory_search expanded optional fields ───────


class TestMemorySearchExpandedOptional:
    def _make_tool_parsed(self, data):
        return ParsedResponse(raw=json.dumps(data), parsed_json=data, is_tool_call=True)

    def test_search_within_accepted(self):
        data = {"tool_call": "memory_search", "queries": ["test"], "search_within": "last_results"}
        result = check_tool_format_valid(
            self._make_tool_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert result.passed

    def test_full_text_accepted(self):
        data = {"tool_call": "memory_search", "queries": ["test"], "full_text": True}
        result = check_tool_format_valid(
            self._make_tool_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert result.passed

    def test_keep_memory_turns_accepted(self):
        data = {"tool_call": "memory_search", "queries": ["test"], "keep_memory_turns": [1, 5]}
        result = check_tool_format_valid(
            self._make_tool_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert result.passed

    def test_search_within_turn_ids_accepted(self):
        data = {"tool_call": "memory_search", "queries": ["test"], "search_within_turn_ids": [10, 20]}
        result = check_tool_format_valid(
            self._make_tool_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert result.passed

    def test_all_new_optional_fields_together(self):
        data = {
            "tool_call": "memory_search",
            "queries": ["test"],
            "category": "char:marcus",
            "search_within": "last_results",
            "full_text": True,
            "keep_memory_turns": [1],
            "search_within_turn_ids": [5],
            "before_lines": 3,
            "after_lines": 3,
        }
        result = check_tool_format_valid(
            self._make_tool_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert result.passed


# ── Round 2: game_time range validation ───────────────────


class TestGameTimeRangeValid:
    def test_valid_game_time(self):
        data = {"state_update": {"game_time": {"day": 5, "hour": 14, "minute": 30}}}
        result = check_game_time_range_valid(
            _make_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert result.passed

    def test_day_zero_fails(self):
        data = {"state_update": {"game_time": {"day": 0, "hour": 12, "minute": 0}}}
        result = check_game_time_range_valid(
            _make_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert not result.passed
        assert "day=0" in result.detail

    def test_negative_day_fails(self):
        data = {"state_update": {"game_time": {"day": -1}}}
        result = check_game_time_range_valid(
            _make_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert not result.passed

    def test_hour_24_fails(self):
        data = {"state_update": {"game_time": {"day": 1, "hour": 24}}}
        result = check_game_time_range_valid(
            _make_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert not result.passed
        assert "hour=24" in result.detail

    def test_hour_negative_fails(self):
        data = {"state_update": {"game_time": {"hour": -1}}}
        result = check_game_time_range_valid(
            _make_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert not result.passed

    def test_minute_60_fails(self):
        data = {"state_update": {"game_time": {"minute": 60}}}
        result = check_game_time_range_valid(
            _make_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert not result.passed
        assert "minute=60" in result.detail

    def test_minute_59_passes(self):
        data = {"state_update": {"game_time": {"minute": 59}}}
        result = check_game_time_range_valid(
            _make_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert result.passed

    def test_skipped_on_tool_call(self):
        parsed = ParsedResponse(raw="{}", parsed_json={"tool_call": "memory_search"}, is_tool_call=True)
        result = check_game_time_range_valid(
            parsed, _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert result.passed

    def test_bool_day_rejected(self):
        data = {"state_update": {"game_time": {"day": True}}}
        result = check_game_time_range_valid(
            _make_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert not result.passed
        assert "integer" in result.detail


# ── Round 2: beat narration no dialogue check ─────────────


class TestBeatNarrationNoDialogue:
    def test_narration_beat_with_dialogue_fails(self):
        data = {"scene_output": {"beats": [
            {"type": "narration", "text": 'The wind howled. "I told you not to come here," she snapped.'},
        ]}}
        result = check_beat_narration_no_dialogue(
            _make_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert not result.passed
        assert "quoted dialogue" in result.detail

    def test_narration_beat_no_dialogue_passes(self):
        data = {"scene_output": {"beats": [
            {"type": "narration", "text": "The wind howled across the empty plain."},
        ]}}
        result = check_beat_narration_no_dialogue(
            _make_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert result.passed

    def test_npc_dialogue_beat_with_quotes_passes(self):
        data = {"scene_output": {"beats": [
            {"type": "npc_dialogue", "speaker": "guard", "text": '"Halt! Who goes there?"'},
        ]}}
        result = check_beat_narration_no_dialogue(
            _make_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert result.passed

    def test_npc_dialogue_in_beat_type_values(self):
        assert "npc_dialogue" in BEAT_TYPE_VALUES

    def test_no_scene_output_skipped(self):
        result = check_beat_narration_no_dialogue(
            _make_parsed({}), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert result.passed


# ── Round 2: NPC evolving_personality ─────────────────────


class TestNpcEvolvingPersonality:
    def test_evolving_personality_in_mutable_fields(self):
        assert "evolving_personality" in MUTABLE_FIELDS

    def test_evolving_personality_update_accepted(self):
        state = _make_state()
        state.characters["guard-bob"] = {
            "name": "Bob", "personality": "stern",
            "background": "ex-soldier", "appearance": "tall",
            "speech_style": "clipped", "location": "gate",
            "current_status": "on-duty", "allegiance": "kingdom",
            "relationship": "neutral", "age": "40s", "gender": "cis-male",
        }
        data = {"character_updates": {
            "guard-bob": {"evolving_personality": "warming up to the party"},
        }}
        result = check_npc_update_fields_valid(
            _make_parsed(data), _make_scenario(),
            TurnSpec(action="test"), state, {},
        )
        assert result.passed


# ── Round 2: location priority aliases ────────────────────


class TestLocationPriorityAliases:
    def test_canonical_values_accepted(self):
        for v in ("critical", "scene", "low"):
            data = {"location_updates": {
                "hotel-lobby": {"atmosphere": {"value": "tense", "priority": v}},
            }}
            result = check_location_updates_valid(
                _make_parsed(data), _make_scenario(),
                TurnSpec(action="test"), _make_state(), {},
            )
            assert result.passed, f"canonical priority '{v}' should pass"

    def test_aliases_accepted(self):
        for alias in PRIORITY_ALIASES:
            data = {"location_updates": {
                "hotel-lobby": {"atmosphere": {"value": "tense", "priority": alias}},
            }}
            result = check_location_updates_valid(
                _make_parsed(data), _make_scenario(),
                TurnSpec(action="test"), _make_state(), {},
            )
            assert result.passed, f"alias '{alias}' should pass"

    def test_unknown_priority_rejected(self):
        data = {"location_updates": {
            "hotel-lobby": {"atmosphere": {"value": "tense", "priority": "bogus"}},
        }}
        result = check_location_updates_valid(
            _make_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert not result.passed
        assert "bogus" in result.detail
