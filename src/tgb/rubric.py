"""Rubric-based grading system for LLM output.

Rubrics are creative-writing-style scoring guides with level descriptors (1-5).
They produce scored grades alongside the existing pass/fail checks, evaluated
by a judge model. Some rubrics also attach computed metrics (e.g. cosine
similarity for repetitiveness).

Rubrics are loaded from YAML files — built-in rubrics ship in the rubrics/
directory, and users can supply additional rubric dirs via CLI.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from tgb.config import Scenario, TurnSpec
from tgb.prompt_builder import AccumulatedState
from tgb.response_parser import ParsedResponse
from tgb.similarity import (
    consecutive_similarity,
    max_consecutive_similarity,
    mean_similarity,
)


# ── Data model ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RubricLevel:
    """One level in a rubric's scoring scale."""

    score: int
    label: str
    description: str


@dataclass(frozen=True)
class Rubric:
    """A single grading rubric loaded from YAML."""

    id: str
    name: str
    category: str
    description: str
    levels: list[RubricLevel]  # sorted 5→1
    scope: str = "turn"  # "turn" or "scenario"
    computed_metric: str = ""  # e.g. "cosine_similarity" — attached alongside judge score

    @property
    def max_score(self) -> int:
        return max(l.score for l in self.levels) if self.levels else 5


@dataclass(frozen=True)
class RubricScore:
    """Result of grading one rubric on one turn (or scenario)."""

    rubric_id: str
    rubric_name: str
    category: str
    score: int  # 1-5
    max_score: int
    reason: str
    scope: str  # "turn" or "scenario"
    action_id: str = ""  # which turn (empty for scenario-scope)
    metric_name: str = ""  # e.g. "cosine_similarity"
    metric_value: float | None = None  # computed metric if applicable

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "rubric_id": self.rubric_id,
            "rubric_name": self.rubric_name,
            "category": self.category,
            "score": self.score,
            "max_score": self.max_score,
            "reason": self.reason,
            "scope": self.scope,
        }
        if self.action_id:
            d["action_id"] = self.action_id
        if self.metric_name:
            d["metric_name"] = self.metric_name
            d["metric_value"] = self.metric_value
        return d


# ── Loading ─────────────────────────────────────────────────────────────────


def _parse_rubric(raw: dict[str, Any], filepath: str = "") -> Rubric:
    """Parse a single rubric from a YAML dict."""
    rubric_id = raw.get("id", "")
    if not rubric_id:
        raise ValueError(f"Rubric missing 'id' in {filepath}")

    levels_raw = raw.get("levels", {})
    if not levels_raw or not isinstance(levels_raw, dict):
        raise ValueError(f"Rubric '{rubric_id}' missing or empty 'levels'")

    levels = []
    for score_str, desc in levels_raw.items():
        score = int(score_str)
        if isinstance(desc, dict):
            label = desc.get("label", "")
            description = desc.get("description", "")
        else:
            # Simple format: score: "description"
            label = ""
            description = str(desc)
        levels.append(RubricLevel(score=score, label=label, description=description))
    levels.sort(key=lambda l: l.score, reverse=True)

    return Rubric(
        id=rubric_id,
        name=raw.get("name", rubric_id),
        category=raw.get("category", "general"),
        description=raw.get("description", ""),
        levels=levels,
        scope=raw.get("scope", "turn"),
        computed_metric=raw.get("computed_metric", ""),
    )


def load_rubrics_from_file(path: Path) -> list[Rubric]:
    """Load rubrics from a YAML file (may contain one rubric or a list)."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    if isinstance(raw, list):
        return [_parse_rubric(r, str(path)) for r in raw]
    elif isinstance(raw, dict):
        # Could be a single rubric or a file with a "rubrics" key
        if "rubrics" in raw:
            return [_parse_rubric(r, str(path)) for r in raw["rubrics"]]
        return [_parse_rubric(raw, str(path))]
    else:
        raise ValueError(f"Invalid rubric file format: {path}")


def load_rubrics(dirs: list[str | Path]) -> dict[str, Rubric]:
    """Load all rubrics from a list of directories, keyed by ID."""
    rubrics: dict[str, Rubric] = {}
    for d in dirs:
        d = Path(d)
        if not d.exists():
            continue
        for f in sorted(d.glob("*.yaml")):
            if f.name.startswith("_"):
                continue
            for rubric in load_rubrics_from_file(f):
                rubrics[rubric.id] = rubric
    return rubrics


def builtin_rubric_dir() -> Path:
    """Return path to the built-in rubrics directory."""
    # rubrics/ is at project root, which is 3 levels up from this file
    # src/tgb/rubric.py -> project_root/rubrics/
    return Path(__file__).parent.parent.parent / "rubrics"


# ── Judge grading ───────────────────────────────────────────────────────────

RUBRIC_JUDGE_SYSTEM_PROMPT = (
    "You are grading a text-adventure game engine's JSON output against specific rubrics.\n\n"
    "You will receive:\n"
    "1. Game context: scenario name, setting tone, player action, world state\n"
    "2. The model's raw JSON response (containing narration, state_update, reasoning, etc.)\n"
    "3. One or more rubrics, each with numbered level descriptors (5 = best, 1 = worst)\n\n"
    "## Grading rules\n"
    "- Each rubric level lists SPECIFIC OBSERVABLE MARKERS — things you can point to in the text.\n"
    "- Pick the level whose markers best match what you observe in the response.\n"
    "- If the response shows markers from two adjacent levels, pick the lower one.\n"
    "- You MUST cite specific evidence: quote words/phrases from the narration, name concrete\n"
    "  observations (e.g. 'narration is 4 sentences', 'NPC appears without prior mention',\n"
    "  'player refusal is re-raised in paragraph 2'). Do NOT use vague language like\n"
    "  'well-written' or 'feels natural'.\n"
    "- A score of 3 means the response is functional/adequate, not that it failed.\n"
    "- A score of 5 requires ALL markers for that level to be present, not just most.\n"
    "- A score of 1 requires at least one critical violation listed at that level.\n\n"
    "Return ONLY valid JSON:\n"
    '{"grades": [{"rubric_id": "...", "score": N, '
    '"reason": "1-2 sentences citing specific text evidence"}]}\n'
)


class RubricGrader:
    """Grades model output against rubrics using a judge LLM."""

    def __init__(self, client: Any, temperature: float = 0.1) -> None:
        self._client = client
        self._temperature = temperature

    def grade_turn(
        self,
        rubrics: list[Rubric],
        parsed: ParsedResponse,
        scenario: Scenario,
        turn: TurnSpec,
        state: AccumulatedState,
        narrations: list[str],  # all narrations so far including this one
    ) -> list[RubricScore]:
        """Grade a single turn against turn-scope rubrics."""
        turn_rubrics = [r for r in rubrics if r.scope == "turn"]
        if not turn_rubrics:
            return []

        scores: list[RubricScore] = []

        # Separate rubrics that need judge vs. computed-only
        judge_rubrics = [r for r in turn_rubrics if not r.computed_metric or r.computed_metric != "cosine_similarity_only"]
        computed_only = [r for r in turn_rubrics if r.computed_metric == "cosine_similarity_only"]

        # Judge-graded rubrics (skip if no client)
        if judge_rubrics and self._client is not None:
            judge_scores = self._judge_grade(judge_rubrics, parsed, scenario, turn, state)
            # Attach computed metrics to judge scores where applicable
            for js in judge_scores:
                rubric = next((r for r in judge_rubrics if r.id == js.rubric_id), None)
                if rubric and rubric.computed_metric == "cosine_similarity":
                    metric_val = self._compute_similarity_metric(narrations)
                    scores.append(RubricScore(
                        rubric_id=js.rubric_id,
                        rubric_name=js.rubric_name,
                        category=js.category,
                        score=js.score,
                        max_score=js.max_score,
                        reason=js.reason,
                        scope=js.scope,
                        action_id=turn.action_id,
                        metric_name="cosine_similarity",
                        metric_value=metric_val,
                    ))
                else:
                    scores.append(RubricScore(
                        rubric_id=js.rubric_id,
                        rubric_name=js.rubric_name,
                        category=js.category,
                        score=js.score,
                        max_score=js.max_score,
                        reason=js.reason,
                        scope=js.scope,
                        action_id=turn.action_id,
                    ))

        # Computed-only rubrics (no judge call needed)
        for rubric in computed_only:
            metric_val = self._compute_similarity_metric(narrations)
            score = self._similarity_to_score(metric_val, rubric)
            scores.append(RubricScore(
                rubric_id=rubric.id,
                rubric_name=rubric.name,
                category=rubric.category,
                score=score,
                max_score=rubric.max_score,
                reason=f"Cosine similarity: {metric_val:.3f}",
                scope="turn",
                action_id=turn.action_id,
                metric_name="cosine_similarity",
                metric_value=metric_val,
            ))

        return scores

    def grade_scenario(
        self,
        rubrics: list[Rubric],
        narrations: list[str],
        scenario: Scenario,
        state: AccumulatedState,
    ) -> list[RubricScore]:
        """Grade scenario-scope rubrics (evaluated across all turns)."""
        scenario_rubrics = [r for r in rubrics if r.scope == "scenario"]
        if not scenario_rubrics:
            return []

        scores: list[RubricScore] = []
        for rubric in scenario_rubrics:
            metric_val = None
            metric_name = ""

            if rubric.computed_metric in ("cosine_similarity", "cosine_similarity_only"):
                metric_val = self._compute_similarity_metric(narrations)
                metric_name = "cosine_similarity"

                if rubric.computed_metric == "cosine_similarity_only":
                    score = self._similarity_to_score(metric_val, rubric)
                    scores.append(RubricScore(
                        rubric_id=rubric.id,
                        rubric_name=rubric.name,
                        category=rubric.category,
                        score=score,
                        max_score=rubric.max_score,
                        reason=f"Mean pairwise cosine similarity: {metric_val:.3f}",
                        scope="scenario",
                        metric_name=metric_name,
                        metric_value=metric_val,
                    ))
                    continue

            # Judge-graded scenario rubrics need a client
            if self._client is None:
                continue

            full_context = "\n---\n".join(
                f"Turn {i+1}: {n}" for i, n in enumerate(narrations)
            )
            judge_score = self._judge_grade_scenario(rubric, full_context, scenario, state)
            scores.append(RubricScore(
                rubric_id=rubric.id,
                rubric_name=rubric.name,
                category=rubric.category,
                score=judge_score.score,
                max_score=rubric.max_score,
                reason=judge_score.reason,
                scope="scenario",
                metric_name=metric_name,
                metric_value=metric_val,
            ))

        return scores

    def _judge_grade(
        self,
        rubrics: list[Rubric],
        parsed: ParsedResponse,
        scenario: Scenario,
        turn: TurnSpec,
        state: AccumulatedState,
    ) -> list[RubricScore]:
        """Call judge model to grade turn against rubrics."""
        context = self._build_context(scenario, turn, state)
        rubric_text = self._format_rubrics_for_prompt(rubrics)

        user_prompt = (
            f"## Game Context\n{context}\n\n"
            f"## Model Response\n```json\n{parsed.raw}\n```\n\n"
            f"## Rubrics\n{rubric_text}\n\n"
            "Grade the response against each rubric. Return JSON with a grades array."
        )

        try:
            text, _ = self._client.complete_text(
                system_prompt=RUBRIC_JUDGE_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=self._temperature,
                max_tokens=1024,
                json_mode=True,
            )
            return self._parse_grades(text, rubrics)
        except Exception as e:
            return [
                RubricScore(
                    rubric_id=r.id,
                    rubric_name=r.name,
                    category=r.category,
                    score=0,
                    max_score=r.max_score,
                    reason=f"Judge error: {e}",
                    scope="turn",
                )
                for r in rubrics
            ]

    def _judge_grade_scenario(
        self,
        rubric: Rubric,
        narrations_text: str,
        scenario: Scenario,
        state: AccumulatedState,
    ) -> RubricScore:
        """Call judge model to grade a scenario-scope rubric."""
        rubric_text = self._format_rubrics_for_prompt([rubric])

        user_prompt = (
            f"## Scenario: {scenario.name}\n"
            f"Setting: {state.campaign_state.get('tone', 'not specified')}\n"
            f"Difficulty: {scenario.campaign.difficulty}\n\n"
            f"## All Turn Narrations\n{narrations_text}\n\n"
            f"## Rubric\n{rubric_text}\n\n"
            "Grade the overall scenario output against the rubric. "
            "Consider ALL turns together. Return JSON with a grades array."
        )

        try:
            text, _ = self._client.complete_text(
                system_prompt=RUBRIC_JUDGE_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=self._temperature,
                max_tokens=512,
                json_mode=True,
            )
            grades = self._parse_grades(text, [rubric])
            return grades[0] if grades else RubricScore(
                rubric_id=rubric.id, rubric_name=rubric.name,
                category=rubric.category, score=0, max_score=rubric.max_score,
                reason="Judge returned no grade", scope="scenario",
            )
        except Exception as e:
            return RubricScore(
                rubric_id=rubric.id, rubric_name=rubric.name,
                category=rubric.category, score=0, max_score=rubric.max_score,
                reason=f"Judge error: {e}", scope="scenario",
            )

    def _format_rubrics_for_prompt(self, rubrics: list[Rubric]) -> str:
        """Format rubrics as human-readable text for the judge prompt."""
        parts = []
        for r in rubrics:
            lines = [f"### {r.id}: {r.name}", f"*{r.description}*", ""]
            for level in r.levels:
                label = f" ({level.label})" if level.label else ""
                lines.append(f"**{level.score}{label}:** {level.description}")
            parts.append("\n".join(lines))
        return "\n\n".join(parts)

    def _build_context(self, scenario: Scenario, turn: TurnSpec, state: AccumulatedState) -> str:
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
            char_summaries = []
            for slug, data in list(state.characters.items())[:5]:
                name = data.get("name", slug) if isinstance(data, dict) else slug
                char_summaries.append(name)
            lines.append(f"Known NPCs: {char_summaries}")
        return "\n".join(lines)

    def _parse_grades(self, text: str, rubrics: list[Rubric]) -> list[RubricScore]:
        """Parse judge response into RubricScore list."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end > start:
                data = json.loads(text[start:end + 1])
            else:
                return []

        grades = data.get("grades", [])
        rubric_map = {r.id: r for r in rubrics}
        results = []

        for grade in grades:
            if not isinstance(grade, dict):
                continue
            rid = grade.get("rubric_id", "")
            rubric = rubric_map.get(rid)
            if not rubric:
                continue
            score = int(grade.get("score", 0))
            score = max(1, min(rubric.max_score, score))
            results.append(RubricScore(
                rubric_id=rid,
                rubric_name=rubric.name,
                category=rubric.category,
                score=score,
                max_score=rubric.max_score,
                reason=grade.get("reason", ""),
                scope=rubric.scope,
            ))

        # Fill in any rubrics that the judge didn't return
        returned_ids = {r.rubric_id for r in results}
        for rubric in rubrics:
            if rubric.id not in returned_ids:
                results.append(RubricScore(
                    rubric_id=rubric.id,
                    rubric_name=rubric.name,
                    category=rubric.category,
                    score=0,
                    max_score=rubric.max_score,
                    reason="Judge did not return grade for this rubric",
                    scope=rubric.scope,
                ))

        return results

    def _compute_similarity_metric(self, narrations: list[str]) -> float:
        """Compute mean pairwise cosine similarity across narrations."""
        clean = [n for n in narrations if n and n.strip()]
        if len(clean) < 2:
            return 0.0
        return mean_similarity(clean)

    def _similarity_to_score(self, sim: float, rubric: Rubric) -> int:
        """Map cosine similarity value to a rubric score.

        Lower similarity = better (more varied output).
        Thresholds:
            sim <= 0.15 → 5 (highly varied)
            sim <= 0.30 → 4 (good variety)
            sim <= 0.45 → 3 (acceptable)
            sim <= 0.60 → 2 (repetitive)
            sim >  0.60 → 1 (very repetitive)
        """
        if sim <= 0.15:
            return 5
        elif sim <= 0.30:
            return 4
        elif sim <= 0.45:
            return 3
        elif sim <= 0.60:
            return 2
        else:
            return 1
