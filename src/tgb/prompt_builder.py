"""Builds (system_prompt, user_prompt) from scenario YAML dataclasses.

Mirrors ZorkEmulator.build_prompt() but reads from benchmark config
instead of DB-backed Campaign/Player objects.
"""

from __future__ import annotations

import json
import re
from typing import Any

from tgb.config import CampaignSetup, PlayerSetup, Scenario, TurnSpec


class AccumulatedState:
    """Mutable state tracker across turns within a scenario run."""

    def __init__(self, scenario: Scenario) -> None:
        self.campaign_state: dict[str, Any] = dict(scenario.campaign.state)
        self.player_state: dict[str, Any] = dict(scenario.player.state)
        self.characters: dict[str, dict[str, Any]] = dict(scenario.campaign.characters)
        self.summary: str = scenario.campaign.summary
        self.last_narration: str = scenario.campaign.last_narration
        self.recent_turns: list[dict[str, Any]] = list(scenario.recent_turns)
        self.turn_number: int = 0
        # Subplot tracking
        self.plot_threads: dict[str, dict[str, Any]] = {}
        self.chapters: dict[str, dict[str, Any]] = {}
        self.consequences: dict[str, dict[str, Any]] = {}
        # History of all tool calls for cross-turn analysis
        self.tool_call_history: list[dict[str, Any]] = []
        # Privacy / visibility tracking
        self.visibility_history: list[dict[str, Any]] = []

    def apply(self, parsed_json: dict[str, Any] | None) -> None:
        """Apply a parsed model response to update accumulated state."""
        if not parsed_json:
            return
        self.turn_number += 1

        # Track tool calls for subplot analysis
        tool_call = parsed_json.get("tool_call", "")
        if tool_call:
            self.tool_call_history.append({
                "turn": self.turn_number,
                "tool_call": tool_call,
                "data": parsed_json,
            })
            self._apply_tool_call(parsed_json)

        # Apply state_update
        state_update = parsed_json.get("state_update")
        if isinstance(state_update, dict):
            self._apply_state_patch(self.campaign_state, state_update)

        # Apply player_state_update
        player_update = parsed_json.get("player_state_update")
        if isinstance(player_update, dict):
            self._apply_state_patch(self.player_state, player_update)

        # Apply character_updates
        char_updates = parsed_json.get("character_updates")
        if isinstance(char_updates, dict):
            for slug, data in char_updates.items():
                if data is None or (isinstance(data, dict) and data.get("remove")):
                    self.characters.pop(slug, None)
                elif isinstance(data, dict):
                    if slug in self.characters:
                        self.characters[slug].update(data)
                    else:
                        self.characters[slug] = dict(data)

        # Update summary
        summary_update = parsed_json.get("summary_update")
        if summary_update:
            self.summary = f"{self.summary} {summary_update}".strip()

        # Track narration for recap detection
        narration = parsed_json.get("narration", "")
        if narration:
            self.last_narration = narration

        # Track turn visibility
        turn_visibility = parsed_json.get("turn_visibility")
        if isinstance(turn_visibility, dict):
            self.visibility_history.append({
                "turn": self.turn_number,
                "visibility": turn_visibility,
            })

    def _apply_tool_call(self, data: dict[str, Any]) -> None:
        """Track subplot tool calls (plot_plan, chapter_plan, consequence_log)."""
        tool = data.get("tool_call", "")

        if tool == "plot_plan":
            plans = data.get("plans", [])
            if isinstance(plans, list):
                for plan in plans:
                    if isinstance(plan, dict):
                        thread = plan.get("thread", "")
                        if thread:
                            if thread in self.plot_threads:
                                self.plot_threads[thread].update(plan)
                            else:
                                plan.setdefault("created_turn", self.turn_number)
                                self.plot_threads[thread] = dict(plan)
                            self.plot_threads[thread]["updated_turn"] = self.turn_number

        elif tool == "chapter_plan":
            action = data.get("action", "")
            chapter_data = data.get("chapter", {})
            if isinstance(chapter_data, str):
                # advance_scene / resolve with slug string
                slug = chapter_data
                if slug in self.chapters:
                    if action == "advance_scene":
                        self.chapters[slug]["current_scene"] = data.get("to_scene", "")
                    elif action == "resolve":
                        self.chapters[slug]["status"] = "resolved"
                        self.chapters[slug]["resolution"] = data.get("resolution", "")
                    self.chapters[slug]["updated_turn"] = self.turn_number
            elif isinstance(chapter_data, dict):
                slug = chapter_data.get("slug", "")
                if slug:
                    if action == "create":
                        chapter_data.setdefault("created_turn", self.turn_number)
                        chapter_data.setdefault("status", "active")
                        self.chapters[slug] = dict(chapter_data)
                    elif slug in self.chapters:
                        self.chapters[slug].update(chapter_data)
                    else:
                        self.chapters[slug] = dict(chapter_data)
                    self.chapters[slug]["updated_turn"] = self.turn_number

        elif tool == "consequence_log":
            adds = data.get("add", [])
            if isinstance(adds, dict):
                adds = [adds]
            if isinstance(adds, list):
                for item in adds:
                    if isinstance(item, dict):
                        cid = item.get("id", item.get("trigger", ""))
                        if cid:
                            item.setdefault("created_turn", self.turn_number)
                            item.setdefault("status", "active")
                            expires_turns = item.get("expires_turns", 0)
                            if isinstance(expires_turns, (int, float)) and expires_turns > 0:
                                item["expires_at_turn"] = self.turn_number + int(expires_turns)
                            self.consequences[cid] = dict(item)
                            self.consequences[cid]["updated_turn"] = self.turn_number

            resolves = data.get("resolve", [])
            if isinstance(resolves, dict):
                resolves = [resolves]
            if isinstance(resolves, list):
                for item in resolves:
                    if isinstance(item, dict):
                        cid = item.get("id", "")
                        if cid and cid in self.consequences:
                            self.consequences[cid]["status"] = "resolved"
                            self.consequences[cid]["resolution"] = item.get("resolution", "")
                            self.consequences[cid]["updated_turn"] = self.turn_number

            removes = data.get("remove", [])
            if isinstance(removes, list):
                for cid in removes:
                    self.consequences.pop(str(cid), None)

    def _apply_state_patch(self, target: dict[str, Any], patch: dict[str, Any]) -> None:
        """Apply a state patch: null values prune keys, dicts merge recursively."""
        for key, val in patch.items():
            if val is None:
                target.pop(key, None)
            elif isinstance(val, dict) and isinstance(target.get(key), dict):
                self._apply_state_patch(target[key], val)
            else:
                target[key] = val


def _player_slug_key(value: object) -> str:
    """Convert a name to a stable kebab-case slug (mirrors ZorkEmulator._player_slug_key)."""
    text = str(value or "").strip().lower()
    if not text:
        return ""
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-")[:64]


def _dump_json(obj: Any) -> str:
    """Compact JSON serialization."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


class PromptBuilder:
    """Constructs system and user prompts matching ZorkEmulator.build_prompt() format."""

    @staticmethod
    def _effective_visibility_default(
        requested_default: str,
        player_state: dict[str, Any],
    ) -> str:
        """Compute effective TURN_VISIBILITY_DEFAULT.

        Mirrors ZorkEmulator._default_prompt_turn_visibility:
        - "private" stays "private"
        - Otherwise: "local" if player has a concrete location, else "public"
        """
        clean = str(requested_default or "").strip().lower()
        if clean == "private":
            return "private"
        # Check if player has a concrete location
        for key in ("room_id", "location", "room_title", "room_summary"):
            raw = str(player_state.get(key) or "").strip()
            if raw:
                return "local"
        return "public"

    def build(
        self,
        scenario: Scenario,
        turn: TurnSpec,
        state: AccumulatedState,
    ) -> tuple[str, str]:
        """Build (system_prompt, user_prompt) for a single turn."""
        from text_game_engine.zork_emulator import ZorkEmulator

        campaign = scenario.campaign
        player = scenario.player

        # Build game_time from state or default
        game_time = state.campaign_state.get("game_time", {
            "day": 1, "hour": 8, "minute": 0,
            "period": "morning", "date_label": "Day 1, Morning",
        })

        # Build model state (campaign state minus internal keys)
        model_state = {k: v for k, v in state.campaign_state.items()
                       if k not in ("game_time", "guardrails_enabled", "on_rails")}

        # Difficulty
        difficulty = campaign.difficulty
        difficulty_notes = ZorkEmulator.DIFFICULTY_NOTES.get(difficulty, "")
        if difficulty_notes:
            difficulty = f"{difficulty} — {difficulty_notes}"

        # Speed multiplier
        speed_mult = campaign.speed

        # Player card
        player_state_prompt = dict(state.player_state)
        player_card = {
            "level": player.level,
            "xp": player.xp,
            "points_total": 10,
            "points_spent": 0,
            "attributes": {},
            "state": player_state_prompt,
        }

        # Active player location
        active_location = {
            "location": player_state_prompt.get("location", ""),
            "room_title": player_state_prompt.get("room_title", ""),
            "room_summary": player_state_prompt.get("room_summary", ""),
        }

        # Rails context (inventory-related)
        rails_context = {
            "inventory": player_state_prompt.get("inventory", []),
        }

        # Build party snapshot with player_slug
        if scenario.party:
            party_snapshot = []
            for entry in scenario.party:
                entry = dict(entry)
                if "player_slug" not in entry:
                    name = entry.get("character_name", entry.get("name", ""))
                    entry["player_slug"] = _player_slug_key(name)
                party_snapshot.append(entry)
        else:
            char_name = player_state_prompt.get("character_name", "Player")
            party_snapshot = [{
                "discord_mention": f"<@{player.user_id}>",
                "character_name": char_name,
                "player_slug": _player_slug_key(char_name),
                "location": player_state_prompt.get("location", ""),
                "party_status": "main_party",
                "is_actor": True,
            }]

        # Characters for prompt
        characters_prompt = state.characters

        # Recent turns text
        recent_text = ""
        if state.recent_turns:
            lines = []
            for rt in state.recent_turns:
                turn_tag = rt.get("tag", "")
                action_text = rt.get("action", "")
                narration_text = rt.get("narration", "")
                lines.append(f"{turn_tag}\nAction: {action_text}\n{narration_text}")
            recent_text = "\n---\n".join(lines)

        # Source material
        source_docs_prompt = ""
        if scenario.source_material and scenario.source_material.documents:
            docs_info = []
            for doc in scenario.source_material.documents:
                docs_info.append({"key": doc.key, "format": doc.format, "summary": doc.summary})
            source_docs_prompt = f"SOURCE_MATERIAL_DOCS: {_dump_json(docs_info)}\n"

        # Construct user prompt
        memory_enabled = campaign.memory
        visibility_default = self._effective_visibility_default(
            turn.turn_visibility_default, player_state_prompt,
        ) if campaign.multi_player else "public"
        user_prompt = (
            f"CAMPAIGN: {campaign.name}\n"
            f"PLAYER_ID: {player.user_id}\n"
            f"IS_NEW_PLAYER: false\n"
            f"TURN_VISIBILITY_DEFAULT: {visibility_default}\n"
            f"GUARDRAILS_ENABLED: {str(campaign.guardrails).lower()}\n"
            f"RAILS_CONTEXT: {_dump_json(rails_context)}\n"
            f"WORLD_SUMMARY: {state.summary}\n"
            f"WORLD_STATE: {_dump_json(model_state)}\n"
            f"CURRENT_GAME_TIME: {_dump_json(game_time)}\n"
            f"SPEED_MULTIPLIER: {speed_mult}\n"
            f"DIFFICULTY: {difficulty}\n"
            f"ATTENTION_WINDOW_SECONDS: 600\n"
            f"CURRENTLY_ATTENTIVE_PLAYERS: {_dump_json([])}\n"
            f"ACTIVE_PLAYER_LOCATION: {_dump_json(active_location)}\n"
            f"CALENDAR: {_dump_json({})}\n"
            f"CALENDAR_REMINDERS:\n\n"
            f"MEMORY_LOOKUP_ENABLED: {str(memory_enabled).lower()}\n"
        )

        if source_docs_prompt:
            user_prompt += source_docs_prompt

        user_prompt += (
            f"WORLD_CHARACTERS: {_dump_json(characters_prompt)}\n"
            f"PLAYER_CARD: {_dump_json(player_card)}\n"
            f"PARTY_SNAPSHOT: {_dump_json(party_snapshot)}\n"
            f"RECENT_TURNS:\n{recent_text}\n"
        )

        # Inject active subplot context
        active_threads = [
            t for t in state.plot_threads.values()
            if t.get("status", "active") == "active"
        ]
        if active_threads:
            user_prompt += f"ACTIVE_PLOT_THREADS: {_dump_json(active_threads[:10])}\n"

        active_chapters = [
            c for c in state.chapters.values()
            if c.get("status", "active") == "active"
        ]
        if active_chapters:
            user_prompt += f"ACTIVE_CHAPTERS: {_dump_json(active_chapters[:8])}\n"

        active_consequences = [
            c for c in state.consequences.values()
            if c.get("status", "active") == "active"
        ]
        if active_consequences:
            user_prompt += f"ACTIVE_CONSEQUENCES: {_dump_json(active_consequences[:12])}\n"

        user_prompt += f"\nPLAYER_ACTION: {turn.action}\n"

        # Construct system prompt
        system_prompt = ZorkEmulator.SYSTEM_PROMPT

        if campaign.guardrails:
            system_prompt += ZorkEmulator.GUARDRAILS_SYSTEM_PROMPT

        if campaign.on_rails:
            on_rails_prompt = getattr(ZorkEmulator, "ON_RAILS_SYSTEM_PROMPT", "")
            if on_rails_prompt:
                system_prompt += on_rails_prompt

        if memory_enabled:
            system_prompt += ZorkEmulator.MEMORY_TOOL_PROMPT
        else:
            disabled = getattr(ZorkEmulator, "MEMORY_TOOL_DISABLED_PROMPT", "")
            if disabled:
                system_prompt += disabled

        if campaign.timed_events:
            timer_prompt = getattr(ZorkEmulator, "TIMER_TOOL_PROMPT", "")
            if timer_prompt:
                system_prompt += timer_prompt

        calendar_prompt = getattr(ZorkEmulator, "CALENDAR_TOOL_PROMPT", "")
        if calendar_prompt:
            system_prompt += calendar_prompt

        roster_prompt = getattr(ZorkEmulator, "ROSTER_PROMPT", "")
        if roster_prompt:
            system_prompt += roster_prompt

        # Plot thread / chapter / consequence tool prompts
        plot_prompt = getattr(ZorkEmulator, "PLOT_PLAN_PROMPT", "")
        if plot_prompt:
            system_prompt += plot_prompt

        chapter_prompt = getattr(ZorkEmulator, "CHAPTER_PLAN_PROMPT", "")
        if chapter_prompt and not campaign.on_rails:
            system_prompt += chapter_prompt

        consequence_prompt = getattr(ZorkEmulator, "CONSEQUENCE_LOG_PROMPT", "")
        if consequence_prompt:
            system_prompt += consequence_prompt

        return system_prompt, user_prompt
