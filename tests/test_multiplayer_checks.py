"""Tests for multiplayer check functions."""

import pytest

from tgb.checks.multiplayer import (
    check_co_located_slugs_valid,
    check_other_player_updates_valid,
)
from tgb.checks.registry import CHECKS
from tgb.response_parser import ParsedResponse
from tgb.config import Scenario, CampaignSetup, PlayerSetup, TurnSpec
from tgb.prompt_builder import AccumulatedState


def _make_scenario(**kwargs) -> Scenario:
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
        import json
        raw = json.dumps(json_data)
    return ParsedResponse(raw=raw, parsed_json=json_data, is_tool_call=is_tool_call)


class TestCoLocatedSlugsValid:
    def test_skip_when_field_absent(self):
        data = {"narration": "You look around."}
        result = check_co_located_slugs_valid(
            _make_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert result.passed
        assert "skipped" in result.detail

    def test_valid_slugs(self):
        data = {"co_located_player_slugs": ["mara-chen"]}
        result = check_co_located_slugs_valid(
            _make_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert result.passed

    def test_not_a_list(self):
        data = {"co_located_player_slugs": "mara-chen"}
        result = check_co_located_slugs_valid(
            _make_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert not result.passed
        assert "list" in result.detail

    def test_unknown_slug(self):
        data = {"co_located_player_slugs": ["unknown-player"]}
        result = check_co_located_slugs_valid(
            _make_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert not result.passed
        assert "not in known" in result.detail

    def test_acting_player_slug_included(self):
        data = {"co_located_player_slugs": ["rico-vega", "mara-chen"]}
        result = check_co_located_slugs_valid(
            _make_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert not result.passed
        assert "acting player" in result.detail


class TestOtherPlayerUpdatesValid:
    def test_skip_when_field_absent(self):
        data = {"narration": "You look around."}
        result = check_other_player_updates_valid(
            _make_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert result.passed
        assert "skipped" in result.detail

    def test_valid_update(self):
        data = {"other_player_state_updates": {"mara-chen": {"location": "fire escape"}}}
        result = check_other_player_updates_valid(
            _make_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert result.passed

    def test_not_a_dict(self):
        data = {"other_player_state_updates": ["mara-chen"]}
        result = check_other_player_updates_valid(
            _make_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert not result.passed
        assert "dict" in result.detail

    def test_unknown_slug_key(self):
        data = {"other_player_state_updates": {"unknown-player": {"location": "elsewhere"}}}
        result = check_other_player_updates_valid(
            _make_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert not result.passed
        assert "not in known" in result.detail

    def test_acting_player_slug_as_key(self):
        data = {"other_player_state_updates": {"rico-vega": {"location": "elsewhere"}}}
        result = check_other_player_updates_valid(
            _make_parsed(data), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert not result.passed
        assert "acting player" in result.detail


class TestRegistryMultiplayer:
    def test_co_located_registered(self):
        assert "co_located_slugs_valid" in CHECKS

    def test_other_player_updates_registered(self):
        assert "other_player_updates_valid" in CHECKS
