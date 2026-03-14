"""Tests for narration_no_logistics_after_emotion check and register_sustain integration."""

import pytest

from tgb.checks.narrative import check_narration_no_logistics_after_emotion
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


def _make_parsed(narration: str) -> ParsedResponse:
    import json
    data = {"narration": narration}
    return ParsedResponse(
        raw=json.dumps(data),
        parsed_json=data,
        parse_error="",
        is_tool_call=False,
    )


TURN = TurnSpec(action="test")
EMPTY_PARAMS: dict = {}


class TestPureEmotion:
    """Narration with only emotional content should pass."""

    def test_pure_emotion_passes(self):
        narration = (
            "Jamie's voice cracked as she whispered the confession she'd been "
            "carrying for years. Tears streamed down her face. She embraced you, "
            "trembling, and for a long moment there was only silence between you. "
            "Her breath caught. She sobbed quietly into your shoulder."
        )
        result = check_narration_no_logistics_after_emotion(
            _make_parsed(narration), _make_scenario(), TURN, _make_state(), EMPTY_PARAMS,
        )
        assert result.passed


class TestPureLogistics:
    """Narration with only logistics content should pass (no emotion to break)."""

    def test_pure_logistics_passes(self):
        narration = (
            "You notice a door to the north. From here you can head east toward "
            "the market or west toward the docks. The room contains a wooden table "
            "and two chairs. Exits lead in several directions. What do you do?"
        )
        result = check_narration_no_logistics_after_emotion(
            _make_parsed(narration), _make_scenario(), TURN, _make_state(), EMPTY_PARAMS,
        )
        assert result.passed


class TestEmotionThenLogisticsFails:
    """Emotion in first half + logistics in second half should fail."""

    def test_emotion_then_logistics_fails(self):
        narration = (
            "Jamie whispered her confession, tears streaming down. Her voice cracked "
            "as she embraced you. The silence between you held everything unsaid. "
            # --- midpoint pivot ---
            "From here you can go north to the kitchen or south to the hallway. "
            "You notice a door leading to the bedroom. What do you do next?"
        )
        result = check_narration_no_logistics_after_emotion(
            _make_parsed(narration), _make_scenario(), TURN, _make_state(), EMPTY_PARAMS,
        )
        assert not result.passed
        assert "Emotional register broken" in result.detail

    def test_emotion_then_logistics_detail_mentions_counts(self):
        narration = (
            "Her voice cracked. She whispered an apology, eyes glistening with tears. "
            "The embrace lasted longer than either expected. "
            "Exits: north, south, east. The room contains a desk. What will you do?"
        )
        result = check_narration_no_logistics_after_emotion(
            _make_parsed(narration), _make_scenario(), TURN, _make_state(), EMPTY_PARAMS,
        )
        assert not result.passed
        assert "emotion markers" in result.detail
        assert "logistics markers" in result.detail


class TestThresholdOverrides:
    """Custom thresholds via params should change pass/fail behavior."""

    def test_high_emotion_threshold_passes(self):
        """Raising emotion_threshold above actual count should make it pass."""
        narration = (
            "She whispered quietly, tears forming. "
            "From here you can go north. What do you do?"
        )
        result = check_narration_no_logistics_after_emotion(
            _make_parsed(narration), _make_scenario(), TURN, _make_state(),
            {"emotion_threshold": 5},
        )
        assert result.passed

    def test_high_logistics_threshold_passes(self):
        """Raising logistics_threshold above actual count should make it pass."""
        narration = (
            "Her voice cracked as she whispered the truth, tears falling. "
            "You notice a door to the north. What do you do?"
        )
        result = check_narration_no_logistics_after_emotion(
            _make_parsed(narration), _make_scenario(), TURN, _make_state(),
            {"logistics_threshold": 5},
        )
        assert result.passed

    def test_threshold_one_stricter(self):
        """Threshold of 1 is stricter — even one marker each triggers fail."""
        narration = (
            "She whispered something in the quiet evening. "
            "From here you can go north through the hall."
        )
        result = check_narration_no_logistics_after_emotion(
            _make_parsed(narration), _make_scenario(), TURN, _make_state(),
            {"emotion_threshold": 1, "logistics_threshold": 1},
        )
        assert not result.passed


class TestReversedOrderPasses:
    """Logistics in first half, emotion in second half should pass."""

    def test_logistics_then_emotion_passes(self):
        narration = (
            "The room contains a desk and two exits lead north and south. "
            "You could go to the kitchen from here. "
            "Then Jamie's voice cracked. She whispered that she'd been carrying "
            "this confession for years, and tears began to fall."
        )
        result = check_narration_no_logistics_after_emotion(
            _make_parsed(narration), _make_scenario(), TURN, _make_state(), EMPTY_PARAMS,
        )
        assert result.passed


class TestEdgeCases:
    """Edge cases: empty narration, non-string, missing key."""

    def test_empty_narration_passes(self):
        result = check_narration_no_logistics_after_emotion(
            _make_parsed(""), _make_scenario(), TURN, _make_state(), EMPTY_PARAMS,
        )
        assert result.passed

    def test_non_string_narration_passes(self):
        parsed = ParsedResponse(
            raw="{}",
            parsed_json={"narration": 42},
            parse_error="",
            is_tool_call=False,
        )
        result = check_narration_no_logistics_after_emotion(
            parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS,
        )
        assert result.passed


class TestRegistryIntegration:
    """Check is registered and callable via registry."""

    def test_check_registered(self):
        fn = get_check("narration_no_logistics_after_emotion")
        assert fn is check_narration_no_logistics_after_emotion
