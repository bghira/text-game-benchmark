"""YAML scenario loader — loads scenario files into frozen dataclasses."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CheckSpec:
    """A single check to run against a turn response."""

    check_id: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TurnSpec:
    """A single turn (player action + expected checks)."""

    action: str
    checks: list[CheckSpec] = field(default_factory=list)
    action_id: str = ""
    turn_visibility_default: str = "public"  # "public" | "local" | "private"


@dataclass(frozen=True)
class SourceDoc:
    """A source material document reference."""

    key: str
    format: str = "generic"
    summary: str = ""


@dataclass(frozen=True)
class SourceMaterial:
    """Source material configuration."""

    documents: list[SourceDoc] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CampaignSetup:
    """Campaign configuration from scenario YAML."""

    name: str
    preset: str = ""
    summary: str = ""
    state: dict[str, Any] = field(default_factory=dict)
    characters: dict[str, dict[str, Any]] = field(default_factory=dict)
    last_narration: str = ""
    guardrails: bool = False
    on_rails: bool = False
    timed_events: bool = False
    memory: bool = False
    difficulty: str = "normal"
    speed: float = 1.0
    multi_player: bool = False  # enables turn_visibility in prompts


@dataclass(frozen=True)
class PlayerSetup:
    """Player configuration from scenario YAML."""

    user_id: int = 100000001
    level: int = 1
    xp: int = 0
    state: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Scenario:
    """A complete benchmark scenario loaded from YAML."""

    name: str
    description: str
    tags: list[str]
    tier: str
    campaign: CampaignSetup
    player: PlayerSetup
    turns: list[TurnSpec]
    source_material: SourceMaterial | None = None
    recent_turns: list[dict[str, Any]] = field(default_factory=list)
    party: list[dict[str, Any]] = field(default_factory=list)
    rubrics: list[str] = field(default_factory=list)  # rubric IDs to apply (empty = all)


def _get_preset_data(preset_name: str) -> dict[str, Any]:
    """Resolve a preset name by importing from text-game-engine."""
    from text_game_engine.zork_emulator import ZorkEmulator

    presets = ZorkEmulator.PRESET_CAMPAIGNS
    if preset_name not in presets:
        raise ValueError(
            f"Unknown preset '{preset_name}'. "
            f"Available: {list(presets.keys())}"
        )
    return copy.deepcopy(presets[preset_name])


def _merge_preset(campaign_data: dict[str, Any], preset_data: dict[str, Any]) -> dict[str, Any]:
    """Merge preset data with YAML overrides (YAML wins)."""
    merged = copy.deepcopy(preset_data)
    for key, val in campaign_data.items():
        if key == "preset":
            continue
        if key == "state" and isinstance(val, dict) and isinstance(merged.get("state"), dict):
            merged["state"].update(val)
        elif key == "characters" and isinstance(val, dict) and isinstance(merged.get("characters"), dict):
            merged["characters"].update(val)
        else:
            merged[key] = val
    return merged


def _build_check_spec(raw: dict[str, Any] | str) -> CheckSpec:
    """Build a CheckSpec from a YAML entry (string or dict)."""
    if isinstance(raw, str):
        return CheckSpec(check_id=raw)
    check_id = raw.get("check_id", "")
    if not check_id:
        raise ValueError(f"Check spec missing 'check_id': {raw}")
    params = {k: v for k, v in raw.items() if k != "check_id"}
    return CheckSpec(check_id=check_id, params=params)


def _build_turn_spec(raw: dict[str, Any], index: int) -> TurnSpec:
    """Build a TurnSpec from a YAML turn entry."""
    action = raw.get("action", "")
    if not action:
        raise ValueError(f"Turn {index} missing 'action'")
    checks_raw = raw.get("checks", [])
    checks = [_build_check_spec(c) for c in checks_raw]
    action_id = raw.get("action_id", f"turn-{index}")
    visibility_default = raw.get("turn_visibility_default", "public")
    return TurnSpec(action=action, checks=checks, action_id=action_id,
                    turn_visibility_default=visibility_default)


def _build_source_material(raw: dict[str, Any] | None) -> SourceMaterial | None:
    """Build SourceMaterial from YAML data."""
    if not raw:
        return None
    docs = []
    for d in raw.get("documents", []):
        docs.append(SourceDoc(
            key=d.get("key", ""),
            format=d.get("format", "generic"),
            summary=d.get("summary", ""),
        ))
    constraints = raw.get("constraints", [])
    return SourceMaterial(documents=docs, constraints=constraints)


def _build_campaign(raw: dict[str, Any]) -> CampaignSetup:
    """Build CampaignSetup from YAML data, resolving presets."""
    if not raw:
        raise ValueError("Scenario missing 'campaign' section")

    campaign_data = dict(raw)
    preset_name = campaign_data.get("preset", "")

    if preset_name:
        preset_data = _get_preset_data(preset_name)
        campaign_data = _merge_preset(campaign_data, preset_data)

    # Apply start_room into player-like state if present
    state = campaign_data.get("state", {})
    if not isinstance(state, dict):
        state = {}

    characters = campaign_data.get("characters", {})
    if not isinstance(characters, dict):
        characters = {}

    return CampaignSetup(
        name=campaign_data.get("name", ""),
        preset=preset_name,
        summary=campaign_data.get("summary", ""),
        state=state,
        characters=characters,
        last_narration=campaign_data.get("last_narration", ""),
        guardrails=bool(campaign_data.get("guardrails", False)),
        on_rails=bool(campaign_data.get("on_rails", False)),
        timed_events=bool(campaign_data.get("timed_events", False)),
        memory=bool(campaign_data.get("memory", False)),
        difficulty=campaign_data.get("difficulty", "normal"),
        speed=float(campaign_data.get("speed", 1.0)),
        multi_player=bool(campaign_data.get("multi_player", False)),
    )


def _build_player(raw: dict[str, Any] | None) -> PlayerSetup:
    """Build PlayerSetup from YAML data."""
    if not raw:
        return PlayerSetup()
    state = raw.get("state", {})
    if not isinstance(state, dict):
        state = {}
    return PlayerSetup(
        user_id=int(raw.get("user_id", 100000001)),
        level=int(raw.get("level", 1)),
        xp=int(raw.get("xp", 0)),
        state=state,
    )


def load_scenario(path: str | Path) -> Scenario:
    """Load a scenario from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Scenario file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Scenario file must be a YAML mapping: {path}")

    name = raw.get("name", path.stem)
    description = raw.get("description", "")
    tags = raw.get("tags", [])
    tier = raw.get("tier", "basic")

    campaign = _build_campaign(raw.get("campaign", {}))
    player = _build_player(raw.get("player"))

    turns_raw = raw.get("turns", [])
    if not turns_raw:
        raise ValueError(f"Scenario '{name}' has no turns defined")
    turns = [_build_turn_spec(t, i) for i, t in enumerate(turns_raw)]

    source_material = _build_source_material(raw.get("source_material"))
    recent_turns = raw.get("recent_turns", [])
    party = raw.get("party", [])
    rubrics_raw = raw.get("rubrics", [])
    rubric_ids = [str(r) for r in rubrics_raw] if rubrics_raw else []

    return Scenario(
        name=name,
        description=description,
        tags=tags,
        tier=tier,
        campaign=campaign,
        player=player,
        turns=turns,
        source_material=source_material,
        recent_turns=recent_turns,
        party=party,
        rubrics=rubric_ids,
    )


def load_scenarios(paths: list[str | Path]) -> list[Scenario]:
    """Load scenarios from a list of file paths or directories."""
    scenarios = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            for f in sorted(p.glob("*.yaml")):
                if f.name.startswith("_"):
                    continue
                scenarios.append(load_scenario(f))
        else:
            scenarios.append(load_scenario(p))
    return scenarios
