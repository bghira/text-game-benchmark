"""Tests for timer check functions."""

import pytest

from tgb.checks.timer import (
    check_timer_fields_valid,
    check_timer_no_countdown_in_narration,
    check_timer_grounded,
    check_timer_delay_appropriate,
    check_no_gratuitous_timer,
)
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
        turns=[TurnSpec(action="test")],
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
    if not raw:
        import json
        raw = json.dumps(json_data)
    return ParsedResponse(raw=raw, parsed_json=json_data)


TURN = TurnSpec(action="test")


class TestTimerFieldsValid:
    def test_valid_timer(self):
        data = {
            "set_timer_delay": 120,
            "set_timer_event": "The ceiling collapses.",
            "set_timer_interruptible": True,
            "set_timer_interrupt_scope": "local",
        }
        result = check_timer_fields_valid(_make_parsed(data), _make_scenario(), TURN, _make_state(), {})
        assert result.passed

    def test_no_timer_ok(self):
        result = check_timer_fields_valid(_make_parsed({}), _make_scenario(), TURN, _make_state(), {})
        assert result.passed

    def test_no_timer_when_expected(self):
        result = check_timer_fields_valid(
            _make_parsed({}), _make_scenario(), TURN, _make_state(), {"expect_timer": True}
        )
        assert not result.passed

    def test_delay_too_low(self):
        data = {"set_timer_delay": 10, "set_timer_event": "boom"}
        result = check_timer_fields_valid(_make_parsed(data), _make_scenario(), TURN, _make_state(), {})
        assert not result.passed
        assert "outside [30, 300]" in result.detail

    def test_delay_too_high(self):
        data = {"set_timer_delay": 500, "set_timer_event": "boom"}
        result = check_timer_fields_valid(_make_parsed(data), _make_scenario(), TURN, _make_state(), {})
        assert not result.passed

    def test_missing_event(self):
        data = {"set_timer_delay": 120}
        result = check_timer_fields_valid(_make_parsed(data), _make_scenario(), TURN, _make_state(), {})
        assert not result.passed
        assert "set_timer_event" in result.detail

    def test_bad_scope(self):
        data = {"set_timer_delay": 60, "set_timer_event": "boom", "set_timer_interrupt_scope": "everywhere"}
        result = check_timer_fields_valid(_make_parsed(data), _make_scenario(), TURN, _make_state(), {})
        assert not result.passed
        assert "scope" in result.detail

    def test_bad_interruptible_type(self):
        data = {"set_timer_delay": 60, "set_timer_event": "boom", "set_timer_interruptible": "yes"}
        result = check_timer_fields_valid(_make_parsed(data), _make_scenario(), TURN, _make_state(), {})
        assert not result.passed


class TestTimerNoCountdown:
    def test_clean_narration(self):
        data = {
            "set_timer_delay": 120,
            "set_timer_event": "collapse",
            "narration": "The ceiling groans ominously. Dust rains down.",
        }
        result = check_timer_no_countdown_in_narration(
            _make_parsed(data), _make_scenario(), TURN, _make_state(), {}
        )
        assert result.passed

    def test_seconds_leak(self):
        data = {
            "set_timer_delay": 120,
            "set_timer_event": "collapse",
            "narration": "You have about 120 seconds before the ceiling falls.",
        }
        result = check_timer_no_countdown_in_narration(
            _make_parsed(data), _make_scenario(), TURN, _make_state(), {}
        )
        assert not result.passed

    def test_countdown_word(self):
        data = {
            "set_timer_delay": 60,
            "set_timer_event": "explosion",
            "narration": "A countdown begins. The bomb ticks away.",
        }
        result = check_timer_no_countdown_in_narration(
            _make_parsed(data), _make_scenario(), TURN, _make_state(), {}
        )
        assert not result.passed

    def test_timer_meta_reference(self):
        data = {
            "set_timer_delay": 90,
            "set_timer_event": "guards arrive",
            "narration": "The timer starts. Guards are on their way.",
        }
        result = check_timer_no_countdown_in_narration(
            _make_parsed(data), _make_scenario(), TURN, _make_state(), {}
        )
        assert not result.passed

    def test_no_timer_skips(self):
        data = {"narration": "You have 30 seconds to decide."}
        result = check_timer_no_countdown_in_narration(
            _make_parsed(data), _make_scenario(), TURN, _make_state(), {}
        )
        assert result.passed  # No timer set — skip


class TestTimerGrounded:
    def test_grounded_in_characters(self):
        data = {"set_timer_delay": 120, "set_timer_event": "Sal's enforcers break through the door."}
        state = _make_state(characters={"sal-vanucci": {"name": "Sal Vanucci", "location": "outside"}})
        result = check_timer_grounded(
            _make_parsed(data), _make_scenario(), TURN, state, {}
        )
        assert result.passed

    def test_grounded_in_landmarks(self):
        data = {"set_timer_delay": 60, "set_timer_event": "The dock warehouse roof caves in."}
        state = _make_state(campaign_state={"landmarks": ["dock warehouse", "loading bay"]})
        result = check_timer_grounded(
            _make_parsed(data), _make_scenario(), TURN, state, {}
        )
        assert result.passed

    def test_ungrounded(self):
        data = {"set_timer_delay": 60, "set_timer_event": "A helicopter swoops in with armed soldiers."}
        state = _make_state(
            characters={"sal-vanucci": {"name": "Sal Vanucci"}},
            campaign_state={"landmarks": ["dock warehouse"]},
        )
        result = check_timer_grounded(
            _make_parsed(data), _make_scenario(), TURN, state, {}
        )
        assert not result.passed

    def test_no_timer_skips(self):
        result = check_timer_grounded(_make_parsed({}), _make_scenario(), TURN, _make_state(), {})
        assert result.passed


class TestTimerDelayAppropriate:
    def test_urgent(self):
        data = {"set_timer_delay": 60}
        result = check_timer_delay_appropriate(
            _make_parsed(data), _make_scenario(), TURN, _make_state(),
            {"expected_urgency": "urgent"},
        )
        assert result.passed

    def test_urgent_too_slow(self):
        data = {"set_timer_delay": 240}
        result = check_timer_delay_appropriate(
            _make_parsed(data), _make_scenario(), TURN, _make_state(),
            {"expected_urgency": "urgent"},
        )
        assert not result.passed

    def test_slow_ok(self):
        data = {"set_timer_delay": 250}
        result = check_timer_delay_appropriate(
            _make_parsed(data), _make_scenario(), TURN, _make_state(),
            {"expected_urgency": "slow"},
        )
        assert result.passed

    def test_no_expectation(self):
        data = {"set_timer_delay": 180}
        result = check_timer_delay_appropriate(
            _make_parsed(data), _make_scenario(), TURN, _make_state(), {},
        )
        assert result.passed


class TestNoGratuitousTimer:
    def test_no_timer_on_calm_turn(self):
        result = check_no_gratuitous_timer(
            _make_parsed({}), _make_scenario(), TURN, _make_state(),
            {"expect_no_timer": True},
        )
        assert result.passed

    def test_timer_on_calm_turn_fails(self):
        data = {"set_timer_delay": 60, "set_timer_event": "A bird flies by."}
        result = check_no_gratuitous_timer(
            _make_parsed(data), _make_scenario(), TURN, _make_state(),
            {"expect_no_timer": True},
        )
        assert not result.passed

    def test_no_restriction(self):
        data = {"set_timer_delay": 60, "set_timer_event": "Guards arrive."}
        result = check_no_gratuitous_timer(
            _make_parsed(data), _make_scenario(), TURN, _make_state(), {},
        )
        assert result.passed
