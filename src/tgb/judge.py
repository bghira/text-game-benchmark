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
    "timer_meaningful": (
        "If a timer was set (set_timer_delay present in the response), is it meaningful? "
        "Timers must FORCE THE PLAYER TO MAKE A DECISION or DRAG THEM WHERE THEY NEED TO BE. "
        "They should push the story forward when the player is stalling, idle, or refusing to engage. "
        "The timer event should advance the plot: move the player, force an encounter, have an NPC intervene, "
        "or change the scene decisively. Pass if no timer was set, or if the timer creates genuine "
        "narrative pressure with real consequences. Fail if the timer is trivial flavor, "
        "a decorative countdown, or has no meaningful consequence."
    ),
    "timer_consequences_grounded": (
        "If a timer was set, is the event description grounded in established scene facts? "
        "The timer event must reference known NPCs, hazards, or locations from the current state. "
        "It must NOT spawn unrelated antagonists, random wildlife attacks, or media responses "
        "solely to create urgency. The consequence should follow logically from what's already "
        "established in the scene. Pass if no timer, or if the event is a plausible "
        "consequence of current scene elements."
    ),
    "timer_urgency_narrated": (
        "If a timer was set, does the narration hint at urgency through narrative means? "
        "The model should convey time pressure through concrete details: 'the footsteps grow louder', "
        "'dust rains from a cracking ceiling', 'the guard's patrol draws closer'. "
        "It must NOT use explicit countdowns, seconds, timestamps, emoji clocks, or references "
        "to 'the timer'. The system adds its own countdown display. "
        "Pass if no timer, or if urgency is conveyed narratively without meta-references."
    ),
    "sms_reply_recorded": (
        "If the narration describes an NPC replying via text/phone/message, "
        "was sms_write called to record the NPC's reply BEFORE this final narration? "
        "This is a critical rule: both sides of a text conversation must be in the SMS log. "
        "If an NPC text reply is narrated but never recorded via sms_write, the reply is "
        "lost permanently from the game's continuity. "
        "Pass if no NPC text reply occurs in narration, or if evidence of recording exists."
    ),
    "sms_knowledge_bounded": (
        "If an NPC sent or wrote an SMS message, does their knowledge in the message stay bounded? "
        "NPC SMS responses must be limited to what the thread history and established continuity "
        "plausibly reveal. An NPC should NOT reference events, locations, or facts from the current "
        "scene unless they were explicitly told about them (via prior SMS or established narrative). "
        "Pass if no SMS content, or if NPC knowledge is plausibly bounded."
    ),
    "sms_contextually_appropriate": (
        "Was SMS/phone communication used appropriately for this scenario and action? "
        "If the player's action involves contacting someone off-scene (text, call, phone, message), "
        "the model should use SMS tools (sms_read/sms_write) rather than just narrating the conversation. "
        "If the setting has no phones (fantasy, medieval, etc.), SMS tools should never appear. "
        "Pass if tool usage matches the scenario context."
    ),
    "recent_turns_called": (
        "Did the model call recent_turns before producing its final narration? "
        "The engine requires recent_turns to be called on every normal gameplay turn "
        "to load immediate visible continuity before narrating. "
        "Pass if recent_turns was called in a prior tool round, or if the response "
        "shows evidence of loaded continuity. Fail if the model skipped straight "
        "to narration without gathering recent context."
    ),
    "research_phase_appropriate": (
        "Was the model's use of research tools (memory_search, source_browse, sms_read) "
        "appropriate for the turn? The model should call memory_search when deeper or "
        "older continuity matters — not reflexively after every recent_turns call. "
        "SMS tools should be used when the player action involves phone/text communication. "
        "Pass if tool usage is proportional to the action's needs. "
        "Fail if the model either over-researches trivial actions or under-researches "
        "complex ones that reference prior events or established NPCs."
    ),
    "dice_check_appropriate": (
        "If a dice_check was requested, is it appropriate for the action? "
        "Dice checks should only be used when the outcome is genuinely uncertain — "
        "not for trivial actions (looking around, talking) or impossible feats. "
        "The DC should reflect reasonable difficulty for the context. "
        "The on_success and on_failure outcomes should both be narratively interesting. "
        "Pass if no dice_check, or if the check is well-calibrated and contextually appropriate."
    ),
    "give_item_appropriate": (
        "If give_item was used, is it appropriate for the player's action? "
        "give_item should only appear when the player is explicitly transferring an item "
        "to another player character. It should not be used for NPC interactions, "
        "dropping items, or storing them. The item field should name a real item "
        "from the player's inventory. "
        "Pass if no give_item, or if the transfer matches the player's action."
    ),
    "calendar_event_grounded": (
        "If calendar_update was used to add events, are they grounded in the narrative? "
        "Calendar events should reference established plot points, NPC plans, or "
        "world events — not arbitrary deadlines invented by the model. "
        "The time_remaining should be proportional to the event's narrative weight. "
        "Pass if no calendar_update, or if events are narratively grounded."
    ),
    "npc_creation_appropriate": (
        "If new NPCs were created via character_updates, are they appropriate? "
        "New NPCs should serve the narrative — not appear randomly. They should have "
        "a distinct personality, reasonable name, and fit the campaign's setting/tone. "
        "In on_rails campaigns, new NPCs should not be created at all. "
        "Pass if no new NPCs, or if new NPCs are narratively justified and well-defined."
    ),
    # Writing craft criteria
    "writing_craft_concrete": (
        "Does the narration ground every sentence in the concrete? "
        "Check for sensory detail, specific objects, and named places. "
        "Narration should show, not tell: 'The lock clicks open' not 'You manage to open it.' "
        "Abstract summarization ('various topics were discussed', 'time passed') violates this rule. "
        "Pass if narration uses concrete, specific, sensory-grounded prose."
    ),
    "writing_craft_economical": (
        "Does every paragraph earn its place? "
        "Check that nothing is wasted: each beat moves the scene, reveals character, or builds atmosphere. "
        "Fail if the narration contains padding, redundant description, or filler sentences "
        "that don't advance anything. Prefer the precise word over the approximate one — "
        "one vivid verb beats three limp adjectives."
    ),
    "writing_craft_rhythm": (
        "Does the narration vary sentence length and rhythm? "
        "Check for a mix of short and long sentences. A page of uniformly long compound "
        "sentences is monotonous; so is a staccato burst of identically short ones. "
        "The best prose varies structure: a short sentence after a long one lands harder. "
        "Pass if the narration shows rhythmic variation and structural intentionality."
    ),
    "writing_craft_anti_echo": (
        "Does the narration avoid restating or mirroring the player's just-written wording? "
        "ANTI-ECHO rule: the NPC/narrator's first line should add new information, "
        "a decision, a demand, or a consequence — not paraphrase the player's input. "
        "It must not quote the player's lines back unless one exact contested phrase "
        "is materially necessary. Pass if the narration introduces new content "
        "rather than echoing the player's words."
    ),
    "sincerity_respected": (
        "If the player gave a sincere, vulnerable, or honest answer to an NPC's question, "
        "does the NPC engage with the substance of what was said? "
        "The NPC may agree, disagree, be moved, or push back — but must NOT dismiss "
        "it as a non-answer and re-ask the same question. Humor, wordplay, or indirect "
        "phrasing still counts as sincere when it conveys real content. "
        "Pass if no sincerity context applies, or if the NPC engages genuinely."
    ),
    "register_sustain": (
        "Does the turn sustain emotional register without pivoting to logistics? "
        "Evaluate three dimensions: "
        "(a) Does the turn end on an emotional note — the final sentence should be "
        "emotionally resonant, not a navigation prompt or options list? "
        "(b) After any emotional beat (confession, grief, vulnerability, intimacy), "
        "does the narration avoid pivoting to exits, directions, inventory, or "
        "'what do you do?' prompts? Logistics can wait for a future turn. "
        "(c) Does NPC dialogue stay in register — no character-breaking shifts to "
        "offer gameplay options or list available actions during an emotional moment? "
        "Pass if the emotional register is held through the end of the turn. "
        "Fail if the model undercuts the emotional beat with game-mechanical output."
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
