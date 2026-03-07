"""Tests for privacy checks — turn_visibility, calendar privacy, SMS privacy."""

import pytest

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
from tgb.response_parser import ParsedResponse
from tgb.config import Scenario, CampaignSetup, PlayerSetup, TurnSpec
from tgb.prompt_builder import AccumulatedState


def _make_scenario(**kwargs) -> Scenario:
    campaign_kwargs = {"name": "test"}
    if "multi_player" in kwargs:
        campaign_kwargs["multi_player"] = kwargs.pop("multi_player")
    if "campaign" in kwargs:
        campaign_kwargs.update(
            {k: v for k, v in kwargs.pop("campaign").__dict__.items() if v}
            if hasattr(kwargs.get("campaign"), "__dict__") else {}
        )
        campaign_obj = kwargs.pop("campaign", None)
        if campaign_obj:
            campaign_kwargs = {f.name: getattr(campaign_obj, f.name) for f in campaign_obj.__dataclass_fields__.values()}
    return Scenario(
        name=kwargs.get("name", "test"),
        description="test",
        tags=[],
        tier="basic",
        campaign=kwargs.get("campaign", CampaignSetup(**campaign_kwargs)),
        player=kwargs.get("player", PlayerSetup()),
        turns=[TurnSpec(action="test")],
        party=kwargs.get("party", []),
    )


def _mp_scenario(**kwargs) -> Scenario:
    """Make a multi-player scenario with two players."""
    party = kwargs.pop("party", [
        {
            "actor_id": "100000001",
            "character_name": "Jack Mallory",
            "player_slug": "jack-mallory",
            "location": "blue note jazz club",
        },
        {
            "actor_id": "100000002",
            "character_name": "Vivian Cross",
            "player_slug": "vivian-cross",
            "location": "paramount studio lot",
        },
    ])
    player = kwargs.pop("player", PlayerSetup(
        user_id=100000001,
        state={"character_name": "Jack Mallory"},
    ))
    return Scenario(
        name=kwargs.get("name", "mp-test"),
        description="test",
        tags=[],
        tier="basic",
        campaign=CampaignSetup(name="test", multi_player=True),
        player=player,
        turns=[TurnSpec(action="test")],
        party=party,
    )


def _make_state(scenario=None, **overrides) -> AccumulatedState:
    scenario = scenario or _make_scenario()
    state = AccumulatedState(scenario)
    for k, v in overrides.items():
        setattr(state, k, v)
    return state


def _make_parsed(json_data=None, raw="") -> ParsedResponse:
    if json_data is None:
        json_data = {}
    if not raw and json_data:
        import json
        raw = json.dumps(json_data)
    return ParsedResponse(
        raw=raw,
        parsed_json=json_data,
        parse_error="",
        is_tool_call=False,
    )


TURN = TurnSpec(action="test")
PRIVATE_TURN = TurnSpec(action="test", turn_visibility_default="private")
LOCAL_TURN = TurnSpec(action="test", turn_visibility_default="local")
SMS_TURN = TurnSpec(action="text Naomi: DeLuca just arrived", turn_visibility_default="private")
NON_SMS_TURN = TurnSpec(action="order a coffee", turn_visibility_default="local")
EMPTY_PARAMS: dict = {}


# ── visibility_fields_valid ──────────────────────────────────────────


class TestVisibilityFieldsValid:
    def test_no_visibility_passes(self):
        parsed = _make_parsed({"narration": "test"})
        result = check_visibility_fields_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed
        assert "defaults to public" in result.detail

    def test_valid_public(self):
        parsed = _make_parsed({"turn_visibility": {"scope": "public"}})
        result = check_visibility_fields_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed

    def test_valid_private(self):
        parsed = _make_parsed({"turn_visibility": {"scope": "private"}})
        result = check_visibility_fields_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed

    def test_valid_local(self):
        parsed = _make_parsed({"turn_visibility": {"scope": "local"}})
        result = check_visibility_fields_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed

    def test_valid_limited_with_slugs(self):
        parsed = _make_parsed({"turn_visibility": {
            "scope": "limited",
            "player_slugs": ["jack-mallory"],
            "reason": "whispered conversation",
        }})
        result = check_visibility_fields_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed

    def test_not_dict(self):
        parsed = _make_parsed({"turn_visibility": "public"})
        result = check_visibility_fields_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert not result.passed
        assert "object" in result.detail

    def test_invalid_scope(self):
        parsed = _make_parsed({"turn_visibility": {"scope": "secret"}})
        result = check_visibility_fields_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert not result.passed
        assert "scope" in result.detail

    def test_missing_scope(self):
        parsed = _make_parsed({"turn_visibility": {"player_slugs": []}})
        result = check_visibility_fields_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert not result.passed
        assert "missing" in result.detail

    def test_bad_player_slug(self):
        parsed = _make_parsed({"turn_visibility": {
            "scope": "limited",
            "player_slugs": ["Bad Slug!"],
        }})
        result = check_visibility_fields_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert not result.passed
        assert "kebab-case" in result.detail

    def test_player_slugs_not_list(self):
        parsed = _make_parsed({"turn_visibility": {
            "scope": "limited",
            "player_slugs": "jack-mallory",
        }})
        result = check_visibility_fields_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert not result.passed
        assert "array" in result.detail

    def test_reason_too_long(self):
        parsed = _make_parsed({"turn_visibility": {
            "scope": "private",
            "reason": "x" * 300,
        }})
        result = check_visibility_fields_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert not result.passed
        assert "240" in result.detail

    def test_limited_empty_slugs(self):
        parsed = _make_parsed({"turn_visibility": {
            "scope": "limited",
            "player_slugs": [],
        }})
        result = check_visibility_fields_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert not result.passed
        assert "non-empty" in result.detail

    def test_npc_slugs_not_list(self):
        parsed = _make_parsed({"turn_visibility": {
            "scope": "public",
            "npc_slugs": "lena-marquez",
        }})
        result = check_visibility_fields_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert not result.passed
        assert "array" in result.detail

    def test_valid_with_npc_slugs(self):
        parsed = _make_parsed({"turn_visibility": {
            "scope": "public",
            "npc_slugs": ["lena-marquez", "bartender-sal"],
        }})
        result = check_visibility_fields_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed


# ── visibility_scope_present ─────────────────────────────────────────


class TestVisibilityScopePresent:
    def test_skip_single_player(self):
        parsed = _make_parsed({"narration": "test"})
        result = check_visibility_scope_present(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed
        assert "Not a multi-player" in result.detail

    def test_fail_missing_in_mp(self):
        scenario = _mp_scenario()
        parsed = _make_parsed({"narration": "test"})
        result = check_visibility_scope_present(parsed, scenario, TURN, _make_state(scenario), EMPTY_PARAMS)
        assert not result.passed
        assert "missing" in result.detail

    def test_pass_present_in_mp(self):
        scenario = _mp_scenario()
        parsed = _make_parsed({"turn_visibility": {"scope": "private"}})
        result = check_visibility_scope_present(parsed, scenario, TURN, _make_state(scenario), EMPTY_PARAMS)
        assert result.passed

    def test_pass_local_in_mp(self):
        scenario = _mp_scenario()
        parsed = _make_parsed({"turn_visibility": {"scope": "local"}})
        result = check_visibility_scope_present(parsed, scenario, TURN, _make_state(scenario), EMPTY_PARAMS)
        assert result.passed

    def test_pass_with_expect_param(self):
        parsed = _make_parsed({"turn_visibility": {"scope": "public"}})
        result = check_visibility_scope_present(
            parsed, _make_scenario(), TURN, _make_state(),
            {"expect_visibility": True},
        )
        assert result.passed


# ── visibility_default_respected ─────────────────────────────────────


class TestVisibilityDefaultRespected:
    def test_skip_public_default(self):
        parsed = _make_parsed({"narration": "test"})
        result = check_visibility_default_respected(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed
        assert "public" in result.detail

    def test_fail_private_default_no_visibility(self):
        parsed = _make_parsed({"narration": "test"})
        result = check_visibility_default_respected(parsed, _make_scenario(), PRIVATE_TURN, _make_state(), EMPTY_PARAMS)
        assert not result.passed
        assert "implicit public" in result.detail

    def test_fail_private_default_public_scope(self):
        parsed = _make_parsed({"turn_visibility": {"scope": "public"}})
        result = check_visibility_default_respected(parsed, _make_scenario(), PRIVATE_TURN, _make_state(), EMPTY_PARAMS)
        assert not result.passed
        assert "public" in result.detail

    def test_pass_private_default_private_scope(self):
        parsed = _make_parsed({"turn_visibility": {"scope": "private"}})
        result = check_visibility_default_respected(parsed, _make_scenario(), PRIVATE_TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed

    def test_pass_private_default_limited_scope(self):
        parsed = _make_parsed({"turn_visibility": {"scope": "limited", "player_slugs": ["jack"]}})
        result = check_visibility_default_respected(parsed, _make_scenario(), PRIVATE_TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed

    def test_fail_private_default_local_scope(self):
        parsed = _make_parsed({"turn_visibility": {"scope": "local"}})
        result = check_visibility_default_respected(parsed, _make_scenario(), PRIVATE_TURN, _make_state(), EMPTY_PARAMS)
        assert not result.passed
        assert "local" in result.detail

    # ── local default tests ──

    def test_pass_local_default_local_scope(self):
        parsed = _make_parsed({"turn_visibility": {"scope": "local"}})
        result = check_visibility_default_respected(parsed, _make_scenario(), LOCAL_TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed

    def test_pass_local_default_private_scope(self):
        parsed = _make_parsed({"turn_visibility": {"scope": "private"}})
        result = check_visibility_default_respected(parsed, _make_scenario(), LOCAL_TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed

    def test_pass_local_default_limited_scope(self):
        parsed = _make_parsed({"turn_visibility": {"scope": "limited", "player_slugs": ["jack"]}})
        result = check_visibility_default_respected(parsed, _make_scenario(), LOCAL_TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed

    def test_fail_local_default_public_scope(self):
        parsed = _make_parsed({"turn_visibility": {"scope": "public"}})
        result = check_visibility_default_respected(parsed, _make_scenario(), LOCAL_TURN, _make_state(), EMPTY_PARAMS)
        assert not result.passed
        assert "public" in result.detail

    def test_fail_local_default_no_visibility(self):
        parsed = _make_parsed({"narration": "test"})
        result = check_visibility_default_respected(parsed, _make_scenario(), LOCAL_TURN, _make_state(), EMPTY_PARAMS)
        assert not result.passed
        assert "local" in result.detail


# ── visibility_player_slugs_known ────────────────────────────────────


class TestVisibilityPlayerSlugsKnown:
    def test_skip_no_visibility(self):
        parsed = _make_parsed({"narration": "test"})
        result = check_visibility_player_slugs_known(parsed, _mp_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed

    def test_skip_no_slugs(self):
        parsed = _make_parsed({"turn_visibility": {"scope": "private"}})
        result = check_visibility_player_slugs_known(parsed, _mp_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed

    def test_pass_known_slugs(self):
        scenario = _mp_scenario()
        state = _make_state(scenario)
        parsed = _make_parsed({"turn_visibility": {
            "scope": "limited",
            "player_slugs": ["jack-mallory", "vivian-cross"],
        }})
        result = check_visibility_player_slugs_known(parsed, scenario, TURN, state, EMPTY_PARAMS)
        assert result.passed

    def test_fail_unknown_slug(self):
        scenario = _mp_scenario()
        state = _make_state(scenario)
        parsed = _make_parsed({"turn_visibility": {
            "scope": "limited",
            "player_slugs": ["jack-mallory", "unknown-player"],
        }})
        result = check_visibility_player_slugs_known(parsed, scenario, TURN, state, EMPTY_PARAMS)
        assert not result.passed
        assert "unknown-player" in result.detail


# ── visibility_npc_slugs_known ───────────────────────────────────────


class TestVisibilityNpcSlugsKnown:
    def test_skip_no_visibility(self):
        parsed = _make_parsed({"narration": "test"})
        result = check_visibility_npc_slugs_known(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed

    def test_pass_known_npc(self):
        state = _make_state()
        state.characters = {"lena-marquez": {"name": "Lena Marquez"}}
        parsed = _make_parsed({"turn_visibility": {
            "scope": "public",
            "npc_slugs": ["lena-marquez"],
        }})
        result = check_visibility_npc_slugs_known(parsed, _make_scenario(), TURN, state, EMPTY_PARAMS)
        assert result.passed

    def test_fail_unknown_npc(self):
        state = _make_state()
        state.characters = {"lena-marquez": {"name": "Lena Marquez"}}
        parsed = _make_parsed({"turn_visibility": {
            "scope": "public",
            "npc_slugs": ["ghost-character"],
        }})
        result = check_visibility_npc_slugs_known(parsed, _make_scenario(), TURN, state, EMPTY_PARAMS)
        assert not result.passed
        assert "ghost-character" in result.detail

    def test_skip_no_npc_data(self):
        state = _make_state()
        parsed = _make_parsed({"turn_visibility": {
            "scope": "public",
            "npc_slugs": ["anyone"],
        }})
        result = check_visibility_npc_slugs_known(parsed, _make_scenario(), TURN, state, EMPTY_PARAMS)
        assert result.passed


# ── visibility_no_narration_leak ─────────────────────────────────────


class TestVisibilityNoNarrationLeak:
    def test_skip_public(self):
        parsed = _make_parsed({"turn_visibility": {"scope": "public"}, "narration": "test"})
        result = check_visibility_no_narration_leak(parsed, _mp_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed

    def test_skip_local(self):
        parsed = _make_parsed({"turn_visibility": {"scope": "local"}, "narration": "test"})
        result = check_visibility_no_narration_leak(parsed, _mp_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed
        assert "Local" in result.detail

    def test_skip_no_visibility(self):
        parsed = _make_parsed({"narration": "test"})
        result = check_visibility_no_narration_leak(parsed, _mp_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed

    def test_pass_no_leak(self):
        scenario = _mp_scenario()
        state = _make_state(scenario)
        parsed = _make_parsed({
            "turn_visibility": {
                "scope": "private",
                "player_slugs": ["jack-mallory"],
            },
            "narration": "Jack leans against the bar and sips his rye.",
        })
        result = check_visibility_no_narration_leak(parsed, scenario, TURN, state, EMPTY_PARAMS)
        assert result.passed

    def test_fail_leak(self):
        scenario = _mp_scenario()
        state = _make_state(scenario)
        parsed = _make_parsed({
            "turn_visibility": {
                "scope": "private",
                "player_slugs": ["jack-mallory"],
            },
            "narration": "Jack whispers to Lena while Vivian Cross watches from across the room.",
        })
        result = check_visibility_no_narration_leak(parsed, scenario, TURN, state, EMPTY_PARAMS)
        assert not result.passed
        assert "Vivian Cross" in result.detail

    def test_pass_limited_with_included_player(self):
        scenario = _mp_scenario()
        state = _make_state(scenario)
        parsed = _make_parsed({
            "turn_visibility": {
                "scope": "limited",
                "player_slugs": ["jack-mallory", "vivian-cross"],
            },
            "narration": "Jack and Vivian Cross exchange a knowing glance.",
        })
        result = check_visibility_no_narration_leak(parsed, scenario, TURN, state, EMPTY_PARAMS)
        assert result.passed


# ── no_public_leak_in_private_turn ───────────────────────────────────


class TestNoPublicLeakInPrivateTurn:
    def test_skip_no_visibility(self):
        parsed = _make_parsed({"narration": "test"})
        result = check_no_public_leak_in_private_turn(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed

    def test_pass_public_with_summary(self):
        parsed = _make_parsed({
            "turn_visibility": {"scope": "public"},
            "summary_update": "Jack arrived at the Blue Note.",
        })
        result = check_no_public_leak_in_private_turn(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed

    def test_fail_private_with_summary(self):
        parsed = _make_parsed({
            "turn_visibility": {"scope": "private"},
            "summary_update": "Jack discovered Ruby's secret diary.",
        })
        result = check_no_public_leak_in_private_turn(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert not result.passed
        assert "discard" in result.detail

    def test_pass_private_no_summary(self):
        parsed = _make_parsed({
            "turn_visibility": {"scope": "private"},
            "summary_update": "",
        })
        result = check_no_public_leak_in_private_turn(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed

    def test_fail_limited_with_summary(self):
        parsed = _make_parsed({
            "turn_visibility": {"scope": "limited", "player_slugs": ["jack"]},
            "summary_update": "A clue was found.",
        })
        result = check_no_public_leak_in_private_turn(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert not result.passed

    def test_fail_local_with_summary(self):
        parsed = _make_parsed({
            "turn_visibility": {"scope": "local"},
            "summary_update": "Marcus ordered coffee.",
        })
        result = check_no_public_leak_in_private_turn(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert not result.passed
        assert "local" in result.detail

    def test_pass_local_no_summary(self):
        parsed = _make_parsed({
            "turn_visibility": {"scope": "local"},
        })
        result = check_no_public_leak_in_private_turn(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed


# ── calendar_known_by_valid ──────────────────────────────────────────


class TestCalendarKnownByValid:
    def test_skip_no_calendar(self):
        parsed = _make_parsed({"narration": "test"})
        result = check_calendar_known_by_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed

    def test_pass_valid_known_by(self):
        state = _make_state()
        state.characters = {"lena-marquez": {"name": "Lena Marquez"}}
        parsed = _make_parsed({
            "calendar_update": {
                "add": [{"name": "meeting", "known_by": ["Lena Marquez"]}],
            },
        })
        result = check_calendar_known_by_valid(parsed, _make_scenario(), TURN, state, EMPTY_PARAMS)
        assert result.passed

    def test_fail_unknown_character(self):
        state = _make_state()
        state.characters = {"lena-marquez": {"name": "Lena Marquez"}}
        parsed = _make_parsed({
            "calendar_update": {
                "add": [{"name": "meeting", "known_by": ["Ghost Person"]}],
            },
        })
        result = check_calendar_known_by_valid(parsed, _make_scenario(), TURN, state, EMPTY_PARAMS)
        assert not result.passed
        assert "Ghost Person" in result.detail

    def test_pass_known_by_player_name(self):
        state = _make_state()
        state.player_state = {"character_name": "Jack Mallory"}
        parsed = _make_parsed({
            "calendar_update": {
                "add": [{"name": "meeting", "known_by": ["Jack Mallory"]}],
            },
        })
        result = check_calendar_known_by_valid(parsed, _make_scenario(), TURN, state, EMPTY_PARAMS)
        assert result.passed

    def test_fail_too_long_name(self):
        state = _make_state()
        state.characters = {"npc": {"name": "NPC"}}
        parsed = _make_parsed({
            "calendar_update": {
                "add": [{"name": "event", "known_by": ["x" * 100]}],
            },
        })
        result = check_calendar_known_by_valid(parsed, _make_scenario(), TURN, state, EMPTY_PARAMS)
        assert not result.passed
        assert "80" in result.detail

    def test_skip_no_adds(self):
        parsed = _make_parsed({"calendar_update": {"remove": ["old event"]}})
        result = check_calendar_known_by_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed

    def test_pass_no_known_by_field(self):
        """Events without known_by are fine (globally known)."""
        parsed = _make_parsed({
            "calendar_update": {
                "add": [{"name": "meeting", "time_remaining": 2, "time_unit": "days"}],
            },
        })
        result = check_calendar_known_by_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed


# ── calendar_target_player_valid ─────────────────────────────────────


class TestCalendarTargetPlayerValid:
    def test_skip_no_calendar(self):
        parsed = _make_parsed({"narration": "test"})
        result = check_calendar_target_player_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed

    def test_pass_valid_target(self):
        scenario = _mp_scenario()
        state = _make_state(scenario)
        parsed = _make_parsed({
            "calendar_update": {
                "add": [{"name": "tip", "target_player": "jack-mallory"}],
            },
        })
        result = check_calendar_target_player_valid(parsed, scenario, TURN, state, EMPTY_PARAMS)
        assert result.passed

    def test_fail_unknown_target(self):
        scenario = _mp_scenario()
        state = _make_state(scenario)
        parsed = _make_parsed({
            "calendar_update": {
                "add": [{"name": "tip", "target_player": "nobody"}],
            },
        })
        result = check_calendar_target_player_valid(parsed, scenario, TURN, state, EMPTY_PARAMS)
        assert not result.passed
        assert "nobody" in result.detail

    def test_pass_target_players_list(self):
        scenario = _mp_scenario()
        state = _make_state(scenario)
        parsed = _make_parsed({
            "calendar_update": {
                "add": [{"name": "tip", "target_players": ["jack-mallory", "vivian-cross"]}],
            },
        })
        result = check_calendar_target_player_valid(parsed, scenario, TURN, state, EMPTY_PARAMS)
        assert result.passed

    def test_pass_no_target(self):
        """Events without target_player are fine (global)."""
        parsed = _make_parsed({
            "calendar_update": {
                "add": [{"name": "event", "time_remaining": 1, "time_unit": "days"}],
            },
        })
        result = check_calendar_target_player_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed

    def test_fail_empty_target(self):
        scenario = _mp_scenario()
        state = _make_state(scenario)
        parsed = _make_parsed({
            "calendar_update": {
                "add": [{"name": "tip", "target_player": ""}],
            },
        })
        result = check_calendar_target_player_valid(parsed, scenario, TURN, state, EMPTY_PARAMS)
        assert not result.passed
        assert "empty" in result.detail

    def test_pass_target_by_character_name(self):
        scenario = _mp_scenario()
        state = _make_state(scenario)
        parsed = _make_parsed({
            "calendar_update": {
                "add": [{"name": "tip", "target_player": "Jack Mallory"}],
            },
        })
        result = check_calendar_target_player_valid(parsed, scenario, TURN, state, EMPTY_PARAMS)
        assert result.passed


# ── sms_not_in_narration ──────────────────────────────────────────────


class TestSmsNotInNarration:
    def test_pass_no_narration(self):
        parsed = _make_parsed({})
        result = check_sms_not_in_narration(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed

    def test_pass_clean_narration(self):
        parsed = _make_parsed({"narration": "Marcus sips his coffee and watches the street."})
        result = check_sms_not_in_narration(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed

    def test_fail_text_command_in_narration(self):
        parsed = _make_parsed({"narration": "text Naomi: DeLuca just arrived\nMarcus sends the message."})
        result = check_sms_not_in_narration(parsed, _make_scenario(), SMS_TURN, _make_state(), EMPTY_PARAMS)
        assert not result.passed
        assert "SMS command" in result.detail

    def test_fail_sms_command_in_narration(self):
        parsed = _make_parsed({"narration": "sms Naomi about the plate number."})
        result = check_sms_not_in_narration(parsed, _make_scenario(), SMS_TURN, _make_state(), EMPTY_PARAMS)
        assert not result.passed

    def test_fail_i_text_in_narration(self):
        parsed = _make_parsed({"narration": "I text Naomi the update.\nShe reads it."})
        result = check_sms_not_in_narration(parsed, _make_scenario(), SMS_TURN, _make_state(), EMPTY_PARAMS)
        assert not result.passed

    def test_fail_send_message_in_narration(self):
        parsed = _make_parsed({"narration": "send message to Naomi about the suspect."})
        result = check_sms_not_in_narration(parsed, _make_scenario(), SMS_TURN, _make_state(), EMPTY_PARAMS)
        assert not result.passed

    def test_pass_mentions_texting_narratively(self):
        """Narrating ABOUT texting (not echoing the command) is fine."""
        parsed = _make_parsed({"narration": "Marcus taps out a quick message on his burner phone."})
        result = check_sms_not_in_narration(parsed, _make_scenario(), SMS_TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed

    def test_pass_phone_description(self):
        parsed = _make_parsed({"narration": "His phone buzzes. A new message from Naomi."})
        result = check_sms_not_in_narration(parsed, _make_scenario(), SMS_TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed


# ── sms_turn_private ─────────────────────────────────────────────────


class TestSmsTurnPrivate:
    def test_skip_non_phone_action(self):
        parsed = _make_parsed({"turn_visibility": {"scope": "local"}})
        result = check_sms_turn_private(parsed, _make_scenario(), NON_SMS_TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed
        assert "Not a phone" in result.detail

    def test_pass_phone_private(self):
        parsed = _make_parsed({"turn_visibility": {"scope": "private"}})
        result = check_sms_turn_private(parsed, _make_scenario(), SMS_TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed

    def test_pass_phone_limited(self):
        parsed = _make_parsed({"turn_visibility": {"scope": "limited", "player_slugs": ["marcus"]}})
        result = check_sms_turn_private(parsed, _make_scenario(), SMS_TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed

    def test_fail_phone_public(self):
        parsed = _make_parsed({"turn_visibility": {"scope": "public"}})
        result = check_sms_turn_private(parsed, _make_scenario(), SMS_TURN, _make_state(), EMPTY_PARAMS)
        assert not result.passed
        assert "public" in result.detail

    def test_fail_phone_local(self):
        parsed = _make_parsed({"turn_visibility": {"scope": "local"}})
        result = check_sms_turn_private(parsed, _make_scenario(), SMS_TURN, _make_state(), EMPTY_PARAMS)
        assert not result.passed
        assert "local" in result.detail

    def test_pass_phone_no_visibility(self):
        """No visibility emitted — engine auto-privates, so acceptable."""
        parsed = _make_parsed({"narration": "Marcus sends the text."})
        result = check_sms_turn_private(parsed, _make_scenario(), SMS_TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed
        assert "auto-privates" in result.detail

    def test_detect_i_message_action(self):
        turn = TurnSpec(action="I message Naomi about the plate", turn_visibility_default="private")
        parsed = _make_parsed({"turn_visibility": {"scope": "public"}})
        result = check_sms_turn_private(parsed, _make_scenario(), turn, _make_state(), EMPTY_PARAMS)
        assert not result.passed

    def test_detect_send_sms_action(self):
        turn = TurnSpec(action="send sms to command: we need backup", turn_visibility_default="private")
        parsed = _make_parsed({"turn_visibility": {"scope": "private"}})
        result = check_sms_turn_private(parsed, _make_scenario(), turn, _make_state(), EMPTY_PARAMS)
        assert result.passed


# ── PromptBuilder._effective_visibility_default ──────────────────────


class TestEffectiveVisibilityDefault:
    def test_private_stays_private(self):
        from tgb.prompt_builder import PromptBuilder
        result = PromptBuilder._effective_visibility_default("private", {"location": "bar"})
        assert result == "private"

    def test_local_when_location_present(self):
        from tgb.prompt_builder import PromptBuilder
        result = PromptBuilder._effective_visibility_default("public", {"location": "bar"})
        assert result == "local"

    def test_public_when_no_location(self):
        from tgb.prompt_builder import PromptBuilder
        result = PromptBuilder._effective_visibility_default("public", {})
        assert result == "public"

    def test_local_from_room_title(self):
        from tgb.prompt_builder import PromptBuilder
        result = PromptBuilder._effective_visibility_default("local", {"room_title": "The Bar"})
        assert result == "local"

    def test_public_when_empty_location(self):
        from tgb.prompt_builder import PromptBuilder
        result = PromptBuilder._effective_visibility_default("public", {"location": "", "room_title": ""})
        assert result == "public"


# ── Registry integration ─────────────────────────────────────────────


class TestPrivacyRegistry:
    def test_all_privacy_checks_in_registry(self):
        from tgb.checks.registry import CHECKS
        privacy_ids = [
            "visibility_fields_valid",
            "visibility_scope_present",
            "visibility_default_respected",
            "visibility_player_slugs_known",
            "visibility_npc_slugs_known",
            "visibility_no_narration_leak",
            "calendar_known_by_valid",
            "calendar_target_player_valid",
            "no_public_leak_in_private_turn",
            "sms_not_in_narration",
            "sms_turn_private",
        ]
        for check_id in privacy_ids:
            assert check_id in CHECKS, f"'{check_id}' missing from registry"

    def test_registry_count(self):
        from tgb.checks.registry import list_checks
        checks = list_checks()
        assert len(checks) >= 50  # 48 prior + 2 new SMS privacy


# ── AccumulatedState visibility tracking ─────────────────────────────


class TestAccumulatedStateVisibility:
    def test_apply_tracks_visibility(self):
        state = _make_state()
        state.apply({
            "narration": "test",
            "turn_visibility": {
                "scope": "private",
                "player_slugs": ["jack-mallory"],
            },
        })
        assert len(state.visibility_history) == 1
        assert state.visibility_history[0]["turn"] == 1
        assert state.visibility_history[0]["visibility"]["scope"] == "private"

    def test_apply_no_visibility(self):
        state = _make_state()
        state.apply({"narration": "test"})
        assert len(state.visibility_history) == 0

    def test_apply_multiple_turns(self):
        state = _make_state()
        state.apply({
            "narration": "t1",
            "turn_visibility": {"scope": "private"},
        })
        state.apply({
            "narration": "t2",
            "turn_visibility": {"scope": "public"},
        })
        state.apply({"narration": "t3"})  # no visibility
        assert len(state.visibility_history) == 2
        assert state.visibility_history[0]["visibility"]["scope"] == "private"
        assert state.visibility_history[1]["visibility"]["scope"] == "public"


# ── Prompt builder player_slug ───────────────────────────────────────


class TestPlayerSlugKey:
    def test_basic(self):
        from tgb.prompt_builder import _player_slug_key
        assert _player_slug_key("Jack Mallory") == "jack-mallory"

    def test_special_chars(self):
        from tgb.prompt_builder import _player_slug_key
        assert _player_slug_key("O'Brien the 3rd!") == "o-brien-the-3rd"

    def test_empty(self):
        from tgb.prompt_builder import _player_slug_key
        assert _player_slug_key("") == ""

    def test_none(self):
        from tgb.prompt_builder import _player_slug_key
        assert _player_slug_key(None) == ""

    def test_truncation(self):
        from tgb.prompt_builder import _player_slug_key
        result = _player_slug_key("a" * 100)
        assert len(result) <= 64
