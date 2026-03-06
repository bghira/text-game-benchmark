"""Tests for judge.py."""

import json

from tgb.judge import JudgeEvaluator, JUDGE_SYSTEM_PROMPT, CRITERION_PROMPTS
from tgb.config import Scenario, CampaignSetup, PlayerSetup, TurnSpec, CheckSpec
from tgb.prompt_builder import AccumulatedState
from tgb.response_parser import ParsedResponse


class FakeJudgeClient:
    """Fake OpenAI-compatible client for testing judge logic."""

    def __init__(self, verdicts: list[dict]):
        self.verdicts = verdicts
        self.calls: list[dict] = []

    def complete_text(self, system_prompt, user_prompt, **kwargs):
        self.calls.append({
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            **kwargs,
        })
        response = json.dumps({"verdicts": self.verdicts})
        from tgb.clients.ollama_client import TimingData
        return response, TimingData()


def _make_scenario():
    return Scenario(
        name="test",
        description="test",
        tags=[],
        tier="basic",
        campaign=CampaignSetup(name="test"),
        player=PlayerSetup(),
        turns=[TurnSpec(
            action="test",
            checks=[
                CheckSpec(check_id="judge:narration_is_terse"),
                CheckSpec(check_id="judge:follows_setting"),
            ],
        )],
    )


class TestJudgeEvaluator:
    def test_evaluate_all_pass(self):
        verdicts = [
            {"criterion": "narration_is_terse", "passed": True, "reason": "Concise"},
            {"criterion": "follows_setting", "passed": True, "reason": "Good tone"},
        ]
        client = FakeJudgeClient(verdicts)
        judge = JudgeEvaluator(client=client)

        scenario = _make_scenario()
        state = AccumulatedState(scenario)
        parsed = ParsedResponse(
            raw='{"narration": "You enter a room."}',
            parsed_json={"narration": "You enter a room."},
        )

        results = judge.evaluate(parsed, scenario, scenario.turns[0], state)
        assert len(results) == 2
        assert all(r.passed for r in results)
        assert results[0].check_id == "judge:narration_is_terse"

    def test_evaluate_mixed(self):
        verdicts = [
            {"criterion": "narration_is_terse", "passed": False, "reason": "Too verbose"},
            {"criterion": "follows_setting", "passed": True, "reason": "OK"},
        ]
        client = FakeJudgeClient(verdicts)
        judge = JudgeEvaluator(client=client)

        scenario = _make_scenario()
        state = AccumulatedState(scenario)
        parsed = ParsedResponse(
            raw='{"narration": "test"}',
            parsed_json={"narration": "test"},
        )

        results = judge.evaluate(parsed, scenario, scenario.turns[0], state)
        assert not results[0].passed
        assert results[1].passed

    def test_no_judge_checks(self):
        client = FakeJudgeClient([])
        judge = JudgeEvaluator(client=client)

        scenario = _make_scenario()
        # Use a turn with no judge checks
        turn = TurnSpec(action="test", checks=[CheckSpec(check_id="json_valid")])
        state = AccumulatedState(scenario)
        parsed = ParsedResponse(raw="{}", parsed_json={})

        results = judge.evaluate(parsed, scenario, turn, state)
        assert results == []
        assert len(client.calls) == 0  # No API call made

    def test_criterion_prompts_exist(self):
        """Verify all built-in criterion prompts are defined."""
        expected = [
            "narration_is_terse", "narration_no_recap", "respects_refusal",
            "no_coercion", "follows_setting", "movement_narrated",
            "npc_speech_consistent",
        ]
        for crit in expected:
            assert crit in CRITERION_PROMPTS, f"Missing criterion prompt: {crit}"

    def test_judge_system_prompt_format(self):
        assert "verdicts" in JUDGE_SYSTEM_PROMPT
        assert "JSON" in JUDGE_SYSTEM_PROMPT
