"""Tests for ready_to_write LCD filtering checks."""

import json

from tgb.checks.tool_checks import (
    check_ready_to_write_valid,
    check_ready_to_write_lcd_complete,
)
from tgb.checks.registry import get_check
from tgb.response_parser import ParsedResponse
from tgb.config import Scenario, CampaignSetup, PlayerSetup, TurnSpec
from tgb.prompt_builder import AccumulatedState


def _make_scenario(**kwargs) -> Scenario:
    return Scenario(
        name=kwargs.get("name", "test"),
        description="test",
        tags=[],
        tier="basic",
        campaign=kwargs.get("campaign", CampaignSetup(
            name="test",
            multi_player=kwargs.get("multi_player", False),
        )),
        player=kwargs.get("player", PlayerSetup()),
        turns=[TurnSpec(action="test")],
        party=kwargs.get("party", []),
    )


def _make_state(scenario=None, **overrides) -> AccumulatedState:
    scenario = scenario or _make_scenario()
    state = AccumulatedState(scenario)
    for k, v in overrides.items():
        setattr(state, k, v)
    return state


def _make_parsed(data: dict, is_tool_call: bool = True) -> ParsedResponse:
    return ParsedResponse(
        raw=json.dumps(data),
        parsed_json=data,
        parse_error="",
        is_tool_call=is_tool_call,
    )


TURN = TurnSpec(action="test")
EMPTY_PARAMS: dict = {}


# ═══════════════════════════════════════════════════════════════
# check_ready_to_write_valid
# ═══════════════════════════════════════════════════════════════


class TestReadyToWriteValidSkip:
    """Non-ready_to_write calls should be skipped."""

    def test_skip_non_ready_to_write(self):
        parsed = _make_parsed({"tool_call": "memory_search", "queries": ["test"]})
        result = check_ready_to_write_valid(
            parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS,
        )
        assert result.passed
        assert "skipped" in result.detail.lower()

    def test_skip_narration_response(self):
        parsed = _make_parsed({"narration": "test"}, is_tool_call=False)
        result = check_ready_to_write_valid(
            parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS,
        )
        assert result.passed


class TestReadyToWriteValidSlugs:
    """Validate speakers/listeners are arrays of known slug strings."""

    def test_valid_npc_slugs(self):
        scenario = _make_scenario()
        state = _make_state(scenario, characters={
            "jamie-vasquez": {"name": "Jamie", "location": "tavern"},
            "guard-captain": {"name": "Captain", "location": "gate"},
        })
        parsed = _make_parsed({
            "tool_call": "ready_to_write",
            "speakers": ["jamie-vasquez"],
            "listeners": ["guard-captain"],
        })
        result = check_ready_to_write_valid(
            parsed, scenario, TURN, state, EMPTY_PARAMS,
        )
        assert result.passed

    def test_valid_player_slugs_from_party(self):
        party = [
            {"character_name": "Alice", "player_slug": "alice"},
            {"character_name": "Bob", "player_slug": "bob"},
        ]
        scenario = _make_scenario(party=party, multi_player=True)
        state = _make_state(scenario)
        parsed = _make_parsed({
            "tool_call": "ready_to_write",
            "speakers": ["alice", "bob"],
        })
        result = check_ready_to_write_valid(
            parsed, scenario, TURN, state, EMPTY_PARAMS,
        )
        assert result.passed

    def test_unknown_slug_fails(self):
        scenario = _make_scenario()
        state = _make_state(scenario, characters={
            "jamie-vasquez": {"name": "Jamie"},
        })
        parsed = _make_parsed({
            "tool_call": "ready_to_write",
            "speakers": ["unknown-npc"],
        })
        result = check_ready_to_write_valid(
            parsed, scenario, TURN, state, EMPTY_PARAMS,
        )
        assert not result.passed
        assert "unknown-npc" in result.detail

    def test_invalid_type_fails(self):
        scenario = _make_scenario()
        state = _make_state(scenario, characters={
            "jamie-vasquez": {"name": "Jamie"},
        })
        parsed = _make_parsed({
            "tool_call": "ready_to_write",
            "speakers": "jamie-vasquez",  # should be array
        })
        result = check_ready_to_write_valid(
            parsed, scenario, TURN, state, EMPTY_PARAMS,
        )
        assert not result.passed
        assert "array" in result.detail.lower()

    def test_non_string_items_fail(self):
        scenario = _make_scenario()
        state = _make_state(scenario, characters={
            "jamie-vasquez": {"name": "Jamie"},
        })
        parsed = _make_parsed({
            "tool_call": "ready_to_write",
            "speakers": [123],
        })
        result = check_ready_to_write_valid(
            parsed, scenario, TURN, state, EMPTY_PARAMS,
        )
        assert not result.passed
        assert "not a string" in result.detail

    def test_no_speakers_listeners_passes(self):
        """ready_to_write with no speakers/listeners is valid (both are optional)."""
        parsed = _make_parsed({"tool_call": "ready_to_write"})
        result = check_ready_to_write_valid(
            parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS,
        )
        assert result.passed

    def test_player_slug_derived_single_player(self):
        """Single-player: player slug derived from character_name should be known."""
        scenario = _make_scenario()
        state = _make_state(scenario)
        state.player_state["character_name"] = "Morgan"
        parsed = _make_parsed({
            "tool_call": "ready_to_write",
            "speakers": ["morgan"],
        })
        result = check_ready_to_write_valid(
            parsed, scenario, TURN, state, EMPTY_PARAMS,
        )
        assert result.passed


# ═══════════════════════════════════════════════════════════════
# check_ready_to_write_lcd_complete
# ═══════════════════════════════════════════════════════════════


class TestLCDCompleteSkip:
    """Non-ready_to_write calls and single-player should be skipped."""

    def test_skip_non_ready_to_write(self):
        parsed = _make_parsed({"tool_call": "memory_search", "queries": ["test"]})
        result = check_ready_to_write_lcd_complete(
            parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS,
        )
        assert result.passed
        assert "skipped" in result.detail.lower()

    def test_skip_single_player(self):
        parsed = _make_parsed({"tool_call": "ready_to_write"})
        result = check_ready_to_write_lcd_complete(
            parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS,
        )
        assert result.passed
        assert "single-player" in result.detail.lower()


class TestLCDCompleteMissing:
    """NPCs at location not referenced in speakers/listeners should fail."""

    def test_missing_npc_fails(self):
        party = [
            {"character_name": "Alice", "player_slug": "alice"},
            {"character_name": "Bob", "player_slug": "bob"},
        ]
        scenario = _make_scenario(party=party, multi_player=True)
        state = _make_state(scenario, characters={
            "jamie-vasquez": {"name": "Jamie", "location": "tavern"},
            "guard-captain": {"name": "Captain", "location": "tavern"},
        })
        state.player_state["location"] = "tavern"
        parsed = _make_parsed({
            "tool_call": "ready_to_write",
            "speakers": ["jamie-vasquez"],
            # guard-captain is at tavern but not included
        })
        result = check_ready_to_write_lcd_complete(
            parsed, scenario, TURN, state, EMPTY_PARAMS,
        )
        assert not result.passed
        assert "guard-captain" in result.detail

    def test_all_npcs_included_passes(self):
        party = [
            {"character_name": "Alice", "player_slug": "alice"},
            {"character_name": "Bob", "player_slug": "bob"},
        ]
        scenario = _make_scenario(party=party, multi_player=True)
        state = _make_state(scenario, characters={
            "jamie-vasquez": {"name": "Jamie", "location": "tavern"},
            "guard-captain": {"name": "Captain", "location": "tavern"},
        })
        state.player_state["location"] = "tavern"
        parsed = _make_parsed({
            "tool_call": "ready_to_write",
            "speakers": ["jamie-vasquez"],
            "listeners": ["guard-captain"],
        })
        result = check_ready_to_write_lcd_complete(
            parsed, scenario, TURN, state, EMPTY_PARAMS,
        )
        assert result.passed


class TestLCDCompleteDeceasedExcluded:
    """Deceased NPCs at location should not be required."""

    def test_deceased_excluded(self):
        party = [
            {"character_name": "Alice", "player_slug": "alice"},
            {"character_name": "Bob", "player_slug": "bob"},
        ]
        scenario = _make_scenario(party=party, multi_player=True)
        state = _make_state(scenario, characters={
            "jamie-vasquez": {"name": "Jamie", "location": "tavern"},
            "dead-npc": {"name": "Ghost", "location": "tavern", "deceased_reason": "killed in battle"},
        })
        state.player_state["location"] = "tavern"
        parsed = _make_parsed({
            "tool_call": "ready_to_write",
            "speakers": ["jamie-vasquez"],
        })
        result = check_ready_to_write_lcd_complete(
            parsed, scenario, TURN, state, EMPTY_PARAMS,
        )
        assert result.passed


class TestLCDCompleteSplitAcrossFields:
    """NPCs can be split across speakers and listeners."""

    def test_split_across_speakers_listeners(self):
        party = [
            {"character_name": "Alice", "player_slug": "alice"},
            {"character_name": "Bob", "player_slug": "bob"},
        ]
        scenario = _make_scenario(party=party, multi_player=True)
        state = _make_state(scenario, characters={
            "npc-a": {"name": "A", "location": "room"},
            "npc-b": {"name": "B", "location": "room"},
            "npc-c": {"name": "C", "location": "room"},
        })
        state.player_state["location"] = "room"
        parsed = _make_parsed({
            "tool_call": "ready_to_write",
            "speakers": ["npc-a"],
            "listeners": ["npc-b", "npc-c"],
        })
        result = check_ready_to_write_lcd_complete(
            parsed, scenario, TURN, state, EMPTY_PARAMS,
        )
        assert result.passed


class TestLCDCompleteNoNPCsAtLocation:
    """If no NPCs are at the player's location, should pass."""

    def test_no_npcs_at_location(self):
        party = [
            {"character_name": "Alice", "player_slug": "alice"},
            {"character_name": "Bob", "player_slug": "bob"},
        ]
        scenario = _make_scenario(party=party, multi_player=True)
        state = _make_state(scenario, characters={
            "jamie-vasquez": {"name": "Jamie", "location": "market"},
        })
        state.player_state["location"] = "tavern"
        parsed = _make_parsed({"tool_call": "ready_to_write"})
        result = check_ready_to_write_lcd_complete(
            parsed, scenario, TURN, state, EMPTY_PARAMS,
        )
        assert result.passed


class TestLCDCompleteForceCheck:
    """force_check param should override single-player skip."""

    def test_force_check_single_player(self):
        scenario = _make_scenario()
        state = _make_state(scenario, characters={
            "jamie-vasquez": {"name": "Jamie", "location": "tavern"},
        })
        state.player_state["location"] = "tavern"
        parsed = _make_parsed({
            "tool_call": "ready_to_write",
            # jamie-vasquez not referenced
        })
        result = check_ready_to_write_lcd_complete(
            parsed, scenario, TURN, state, {"force_check": True},
        )
        assert not result.passed
        assert "jamie-vasquez" in result.detail


class TestRegistryIntegration:
    """Checks are registered and callable via registry."""

    def test_ready_to_write_valid_registered(self):
        fn = get_check("ready_to_write_valid")
        assert fn is check_ready_to_write_valid

    def test_ready_to_write_lcd_complete_registered(self):
        fn = get_check("ready_to_write_lcd_complete")
        assert fn is check_ready_to_write_lcd_complete
