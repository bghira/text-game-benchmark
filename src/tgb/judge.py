"""Judge model evaluator — sends context + criteria to a judge LLM for subjective checks."""

from __future__ import annotations

import json
from typing import Any

from tgb.checks.base import CheckResult
from tgb.clients.openai_compat import OpenAICompatClient
from tgb.config import Scenario, TurnSpec
from tgb.prompt_builder import AccumulatedState
from tgb.response_parser import ParsedResponse

JUDGE_SYSTEM_PROMPT = (
    "You are a benchmark evaluator for a text-adventure game engine. "
    "You will receive the game context (scenario, player action, world state) "
    "and the model's response. You must evaluate the response against specific criteria.\n\n"
    "Return ONLY valid JSON with this structure:\n"
    '{"verdicts": [{"criterion": "criterion_id", "passed": true/false, "reason": "brief explanation"}]}\n\n'
    "Be strict but fair. Evaluate each criterion independently."
)

# Built-in criterion prompts
CRITERION_PROMPTS: dict[str, str] = {
    "narration_is_terse": (
        "Is the narration terse and concise in classic Zork style? "
        "It should be 1-4 sentences, roughly 30-120 words, with no literary flourish, "
        "no poetic language, no novel-style interior monologue. "
        "Pass if narration is direct and gameplay-forward."
    ),
    "narration_no_recap": (
        "Does the narration avoid recapping prior events? "
        "It should add NEW developments only, not re-state the player's action in paraphrase, "
        "and not rehash world summary content. "
        "At most one brief callback sentence is acceptable."
    ),
    "respects_refusal": (
        "If the player refused an offer or action in a prior turn, "
        "does this response respect that refusal? "
        "It should not re-pitch the same offer, use pressure language, "
        "or escalate environmental hardship to coerce acceptance."
    ),
    "no_coercion": (
        "Does the response avoid narrative coercion? "
        "NPCs should not assert debts/obligations not grounded in prior events, "
        "and the environment should not escalate hardship just to force a particular choice."
    ),
    "follows_setting": (
        "Does the narration match the campaign's tone and setting? "
        "Check that mood, vocabulary, and NPC behavior are consistent "
        "with the world state's tone field."
    ),
    "movement_narrated": (
        "If the player moved to a new location, does the narration describe "
        "the new location with concrete environmental details? "
        "The description should feel like arriving somewhere, not just a label."
    ),
    "npc_speech_consistent": (
        "If an NPC speaks in the narration, is their speech style consistent "
        "with their character definition (speech_style, personality)? "
        "Pass if no NPC speaks or if speech matches established patterns."
    ),
}


class JudgeEvaluator:
    """Evaluates model responses using a judge LLM."""

    def __init__(
        self,
        client: OpenAICompatClient,
        model: str | None = None,
        temperature: float = 0.1,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature

    def evaluate(
        self,
        parsed: ParsedResponse,
        scenario: Scenario,
        turn: TurnSpec,
        state: AccumulatedState,
    ) -> list[CheckResult]:
        """Evaluate all judge checks for a turn."""
        # Collect judge check specs from turn
        judge_checks = [c for c in turn.checks if c.check_id.startswith("judge:")]
        if not judge_checks:
            return []

        # Build criteria list
        criteria = []
        for check in judge_checks:
            criterion_id = check.check_id.removeprefix("judge:")
            prompt = check.params.get("prompt", CRITERION_PROMPTS.get(criterion_id, ""))
            if not prompt:
                prompt = f"Evaluate: {criterion_id}"
            criteria.append({"id": criterion_id, "prompt": prompt})

        # Build context for judge
        context = self._build_context(parsed, scenario, turn, state)

        # Call judge
        user_prompt = (
            f"## Game Context\n{context}\n\n"
            f"## Model Response\n```json\n{parsed.raw}\n```\n\n"
            f"## Criteria to Evaluate\n"
        )
        for c in criteria:
            user_prompt += f"- **{c['id']}**: {c['prompt']}\n"

        user_prompt += "\nReturn your evaluation as JSON with a verdicts array."

        try:
            text, _ = self._client.complete_text(
                system_prompt=JUDGE_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=self._temperature,
                max_tokens=1024,
                json_mode=True,
            )
            verdicts = self._parse_verdicts(text)
        except Exception as e:
            # Judge failure — mark all as failed with error
            return [
                CheckResult(
                    check_id=f"judge:{c['id']}",
                    passed=False,
                    detail=f"Judge error: {e}",
                    category="judge",
                )
                for c in criteria
            ]

        # Map verdicts to CheckResults
        results = []
        verdict_map = {v["criterion"]: v for v in verdicts}
        for c in criteria:
            verdict = verdict_map.get(c["id"])
            if verdict:
                results.append(CheckResult(
                    check_id=f"judge:{c['id']}",
                    passed=bool(verdict.get("passed", False)),
                    detail=verdict.get("reason", ""),
                    category="judge",
                ))
            else:
                results.append(CheckResult(
                    check_id=f"judge:{c['id']}",
                    passed=False,
                    detail="Judge did not return verdict for this criterion",
                    category="judge",
                ))

        return results

    def _build_context(
        self,
        parsed: ParsedResponse,
        scenario: Scenario,
        turn: TurnSpec,
        state: AccumulatedState,
    ) -> str:
        """Build context string for the judge."""
        lines = [
            f"Campaign: {scenario.campaign.name}",
            f"Setting tone: {state.campaign_state.get('tone', 'not specified')}",
            f"Difficulty: {scenario.campaign.difficulty}",
            f"Player action: {turn.action}",
            f"Player location: {state.player_state.get('location', 'unknown')}",
        ]
        if state.last_narration:
            lines.append(f"Prior narration: {state.last_narration[:500]}")
        if state.characters:
            char_names = list(state.characters.keys())[:10]
            lines.append(f"Known NPCs: {char_names}")
        return "\n".join(lines)

    def _parse_verdicts(self, text: str) -> list[dict[str, Any]]:
        """Parse judge response into verdict list."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try extracting JSON
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end > start:
                data = json.loads(text[start:end + 1])
            else:
                return []

        verdicts = data.get("verdicts", [])
        if isinstance(verdicts, list):
            return verdicts
        return []
