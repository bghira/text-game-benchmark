"""Builds (system_prompt, user_prompt) from scenario YAML dataclasses.

Mirrors ZorkEmulator.build_prompt() but reads from benchmark config
instead of DB-backed Campaign/Player objects.

The engine now uses a three-stage prompt flow:
  1. bootstrap — decides recent_turns receivers, no narration
  2. research  — loads memory/SMS/planning tools, returns ready_to_write
  3. final     — narrates the turn, returns JSON

The benchmark always uses the "final" stage (single-shot), but the
prompt builder tracks the new field ordering and constants for parity.
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
        # NPC awareness: which NPCs have been referenced in aware_npc_slugs
        self.npc_awareness_history: list[dict[str, Any]] = []
        # Autobiography tracking
        self.autobiography_entries: dict[str, list[dict[str, Any]]] = {}
        # Multiplayer: other player states
        self.other_player_states: dict[str, dict[str, Any]] = {}
        if scenario.party:
            for entry in scenario.party:
                slug = entry.get("player_slug", "")
                if slug:
                    self.other_player_states[slug] = dict(entry)

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

        # Track NPC awareness from scene_output beats
        scene_output = parsed_json.get("scene_output")
        if isinstance(scene_output, dict):
            beats = scene_output.get("beats")
            if isinstance(beats, list):
                for beat in beats:
                    if not isinstance(beat, dict):
                        continue
                    slugs = beat.get("aware_npc_slugs")
                    if isinstance(slugs, list) and slugs:
                        valid_slugs = [s for s in slugs if isinstance(s, str)]
                        if valid_slugs:
                            self.npc_awareness_history.append({
                                "turn": self.turn_number,
                                "slugs": valid_slugs,
                                "beat_type": beat.get("type", ""),
                                "speaker": beat.get("speaker", ""),
                            })

        # Track calendar updates in campaign state
        calendar_update = parsed_json.get("calendar_update")
        if isinstance(calendar_update, dict):
            calendar = self.campaign_state.setdefault("calendar", [])
            if isinstance(calendar, list):
                add_list = calendar_update.get("add")
                if isinstance(add_list, list):
                    for event in add_list:
                        if isinstance(event, dict) and event.get("name"):
                            calendar.append(dict(event))
                remove_list = calendar_update.get("remove")
                if isinstance(remove_list, list):
                    remove_names = {str(n).lower() for n in remove_list if n}
                    self.campaign_state["calendar"] = [
                        e for e in calendar
                        if not isinstance(e, dict) or
                        str(e.get("name", "")).lower() not in remove_names
                    ]

        # Track give_item (just record it; engine handles actual transfer)
        give_item = parsed_json.get("give_item")
        if isinstance(give_item, dict):
            item = give_item.get("item")
            if item and isinstance(item, str):
                # Remove from acting player's inventory if present
                inv = self.player_state.get("inventory")
                if isinstance(inv, list):
                    item_lower = item.lower()
                    self.player_state["inventory"] = [
                        i for i in inv
                        if not (isinstance(i, str) and i.lower() == item_lower)
                    ]

        # Track location_updates
        location_updates = parsed_json.get("location_updates")
        if isinstance(location_updates, dict):
            loc_store = self.campaign_state.setdefault("_location_facts", {})
            if not isinstance(loc_store, dict):
                loc_store = {}
                self.campaign_state["_location_facts"] = loc_store
            for loc_slug, loc_data in location_updates.items():
                if loc_data is None:
                    loc_store.pop(str(loc_slug), None)
                elif isinstance(loc_data, dict):
                    existing = loc_store.setdefault(str(loc_slug), {})
                    if isinstance(existing, dict):
                        existing.update(loc_data)

        # Track story_progression
        story_prog = parsed_json.get("story_progression")
        if isinstance(story_prog, dict):
            self.campaign_state["_last_story_progression"] = dict(story_prog)

        # Track inline tool_calls array (sms_write/sms_schedule side-effects)
        inline_tool_calls = parsed_json.get("tool_calls")
        if isinstance(inline_tool_calls, list):
            for tc in inline_tool_calls:
                if isinstance(tc, dict):
                    tc_tool = tc.get("tool_call", "")
                    if tc_tool in ("sms_write", "sms_schedule"):
                        self._apply_sms_write(tc)

        # Track co-located player slugs
        co_located = parsed_json.get("co_located_player_slugs")
        if isinstance(co_located, list):
            self.campaign_state["_co_located_player_slugs"] = co_located

        # Track other player state updates
        other_updates = parsed_json.get("other_player_state_updates")
        if isinstance(other_updates, dict):
            for slug, patch in other_updates.items():
                if isinstance(patch, dict) and isinstance(slug, str):
                    if slug not in self.other_player_states:
                        self.other_player_states[slug] = {}
                    self._apply_state_patch(self.other_player_states[slug], patch)

        # Track SMS state when sms_write/sms_schedule tool calls are made
        if tool_call in ("sms_write", "sms_schedule"):
            self._apply_sms_write(parsed_json)

        # Track timer setting
        timer_delay = parsed_json.get("set_timer_delay")
        if timer_delay is not None:
            self.campaign_state["_active_timer"] = {
                "delay": timer_delay,
                "event": parsed_json.get("set_timer_event", ""),
                "interruptible": parsed_json.get("set_timer_interruptible", True),
                "turn": self.turn_number,
            }

        # Track dice check, puzzle, minigame results
        dice_check = parsed_json.get("dice_check")
        if isinstance(dice_check, dict):
            self.campaign_state["_last_dice_check"] = dict(dice_check)

        puzzle_trigger = parsed_json.get("puzzle_trigger")
        if isinstance(puzzle_trigger, dict):
            self.campaign_state["_active_puzzle"] = dict(puzzle_trigger)

        minigame_challenge = parsed_json.get("minigame_challenge")
        if isinstance(minigame_challenge, dict):
            self.campaign_state["_active_minigame"] = dict(minigame_challenge)

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
                    elif action in ("resolve", "close"):
                        self.chapters[slug]["status"] = "resolved"
                        resolution = str(data.get("resolution") or "").strip()[:260]
                        self.chapters[slug]["resolution"] = resolution
                    self.chapters[slug]["updated_turn"] = self.turn_number
            elif isinstance(chapter_data, dict):
                slug = chapter_data.get("slug", "")
                if slug:
                    if action == "create":
                        chapter_data.setdefault("created_turn", self.turn_number)
                        chapter_data.setdefault("status", "active")
                        self.chapters[slug] = dict(chapter_data)
                    elif action in ("resolve", "close"):
                        if slug in self.chapters:
                            self.chapters[slug]["status"] = "resolved"
                            resolution = str(data.get("resolution") or "").strip()[:260]
                            self.chapters[slug]["resolution"] = resolution
                        else:
                            self.chapters[slug] = dict(chapter_data)
                            self.chapters[slug]["status"] = "resolved"
                    elif slug in self.chapters:
                        self.chapters[slug].update(chapter_data)
                    else:
                        self.chapters[slug] = dict(chapter_data)
                    self.chapters[slug]["updated_turn"] = self.turn_number

        elif tool in ("autobiography_append", "autobiography_update"):
            entries = data.get("entries", [])
            if isinstance(entries, list):
                for entry in entries:
                    if isinstance(entry, dict):
                        slug = entry.get("character", "")
                        if isinstance(slug, str) and slug:
                            char_entries = self.autobiography_entries.setdefault(slug, [])
                            char_entries.append(dict(entry))
                            # Cap at 64 per character (engine MAX_AUTOBIOGRAPHY_RAW_ENTRIES)
                            if len(char_entries) > 64:
                                self.autobiography_entries[slug] = char_entries[-64:]

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

    # ── SMS tracking (mirrors engine's _sms_write) ──────────────

    SMS_MAX_THREADS = 24
    SMS_MAX_MESSAGES_PER_THREAD = 40

    def _apply_sms_write(self, data: dict[str, Any]) -> None:
        """Track SMS messages in campaign state when sms_write/sms_schedule is called.

        Mirrors engine's _sms_write: stores messages in _sms_threads with
        per-thread message limits and global thread limits.
        """
        thread_key = str(data.get("thread", "")).strip().lower()
        if not thread_key:
            return
        sender = str(data.get("from", data.get("sender", ""))).strip()[:80]
        recipient = str(data.get("to", data.get("recipient", ""))).strip()[:80]
        message = str(data.get("message", "")).strip()[:500]
        if not sender or not recipient or not message:
            return

        threads: dict[str, Any] = self.campaign_state.setdefault("_sms_threads", {})
        if not isinstance(threads, dict):
            threads = {}
            self.campaign_state["_sms_threads"] = threads

        # Get or create thread
        thread = threads.setdefault(thread_key, {"label": thread_key, "messages": []})
        if not isinstance(thread.get("messages"), list):
            thread["messages"] = []

        # Get game_time for timestamp
        game_time = self.campaign_state.get("game_time", {})
        seq = self.campaign_state.get("_sms_message_seq", 0) + 1
        self.campaign_state["_sms_message_seq"] = seq

        msg_entry = {
            "from": sender,
            "to": recipient,
            "message": message,
            "day": game_time.get("day", 1) if isinstance(game_time, dict) else 1,
            "hour": game_time.get("hour", 8) if isinstance(game_time, dict) else 8,
            "minute": game_time.get("minute", 0) if isinstance(game_time, dict) else 0,
            "turn_id": self.turn_number,
            "seq": seq,
        }
        thread["messages"].append(msg_entry)

        # Enforce per-thread message limit
        if len(thread["messages"]) > self.SMS_MAX_MESSAGES_PER_THREAD:
            thread["messages"] = thread["messages"][-self.SMS_MAX_MESSAGES_PER_THREAD:]

        # Enforce global thread limit (evict oldest)
        if len(threads) > self.SMS_MAX_THREADS:
            # Find thread with oldest last message
            oldest_key = min(
                threads,
                key=lambda k: (threads[k].get("messages", [{}])[-1].get("seq", 0)
                               if threads[k].get("messages") else 0),
            )
            del threads[oldest_key]

    # Engine auto-deletes keys set to these string values
    _COMPLETED_VALUES = {
        "complete", "completed", "done", "resolved", "finished",
        "concluded", "vacated", "dispersed", "avoided", "departed",
    }

    def _apply_state_patch(self, target: dict[str, Any], patch: dict[str, Any]) -> None:
        """Apply a state patch: null values prune keys, completed-value strings
        prune keys, dicts merge recursively.

        Matches engine's _apply_state_update behavior.
        """
        for key, val in patch.items():
            if val is None:
                target.pop(key, None)
            elif isinstance(val, str) and val.strip().lower() in self._COMPLETED_VALUES:
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
    """Constructs system and user prompts matching ZorkEmulator.build_prompt() format.

    Mirrors the engine's "final" stage prompt assembly, which is the
    single-shot flow used by the benchmark. The field ordering and
    system prompt composition track the engine's three-stage overhaul.
    """

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
        # Mirrors ZorkEmulator.MODEL_STATE_EXCLUDE_KEYS exactly:
        #   ROOM_STATE_KEYS | { engine internal keys }
        _exclude_keys = {
            # ROOM_STATE_KEYS
            "room_title", "room_description", "room_summary",
            "exits", "location", "room_id",
            # Engine internal keys
            "last_narration", "room_scene_images", "scene_image_model",
            "default_persona", "start_room", "story_outline",
            "current_chapter", "current_scene", "setup_phase", "setup_data",
            "speed_multiplier", "difficulty", "game_time", "calendar",
            "_calendar_reminders", "_auto_fix_counters",
            "_memory_search_usage", "_sms_threads", "_sms_read_state",
            "_sms_message_seq", "_turn_time_index", "literary_styles",
            "_active_puzzle", "_puzzle_result",
            "_active_minigame", "_minigame_result",
            "_last_dice_check", "_last_minigame_result",
        }
        model_state = {k: v for k, v in state.campaign_state.items()
                       if k not in _exclude_keys}

        # Difficulty
        difficulty = campaign.difficulty
        difficulty_notes = ZorkEmulator.DIFFICULTY_NOTES.get(difficulty, "")
        if difficulty_notes:
            difficulty = f"{difficulty} — {difficulty_notes}"

        # Speed multiplier
        speed_mult = campaign.speed

        # Player card — exclude keys the engine filters out
        # (inventory goes in RAILS_CONTEXT, room_description/zork_stats internal)
        _player_exclude = {"inventory", "room_description", "zork_stats"}
        player_state_prompt = {
            k: v for k, v in state.player_state.items()
            if k not in _player_exclude
        }
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

        # Rails context (inventory lives here, not in player_state_prompt)
        rails_context = {
            "inventory": state.player_state.get("inventory", []),
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

        # Characters for prompt — ranked by proximity and relevance,
        # mirroring ZorkEmulator._build_characters_for_prompt()
        characters_prompt = self._build_characters_for_prompt(
            state.characters, player_state_prompt, state.recent_turns,
        )

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
        source_payload: dict[str, Any] = {}
        if scenario.source_material and scenario.source_material.documents:
            docs_info = []
            for doc in scenario.source_material.documents:
                docs_info.append({"key": doc.key, "format": doc.format, "summary": doc.summary})
            source_payload = {
                "available": True,
                "docs": docs_info,
                "keys": [],
                "chunk_count": 0,
            }

        # Build story context from chapters in campaign state
        story_context = self._build_story_context(state.campaign_state, campaign.on_rails)

        # Memory enabled — mirrors _memory_lookup_enabled_for_prompt()
        memory_enabled = self._memory_lookup_enabled(
            campaign.memory, state.summary, turn.action,
            source_available=bool(source_payload.get("available")),
        )

        # Visibility default
        visibility_default = self._effective_visibility_default(
            turn.turn_visibility_default, player_state_prompt,
        ) if campaign.multi_player else "public"

        # Response style note — use _turn_stage_note for final stage,
        # falling back to _turn_response_style_note, then manual composition
        _stage_fn = getattr(ZorkEmulator, "_turn_stage_note", None)
        if callable(_stage_fn):
            # Final stage = default stage
            final_stage = getattr(ZorkEmulator, "PROMPT_STAGE_FINAL", "final")
            response_style_note = _stage_fn(campaign.difficulty, final_stage)
        else:
            _style_fn = getattr(ZorkEmulator, "_turn_response_style_note", None)
            if callable(_style_fn):
                response_style_note = _style_fn(campaign.difficulty)
            else:
                response_style_note = getattr(ZorkEmulator, "RESPONSE_STYLE_NOTE", "")
                _diff_fn = getattr(ZorkEmulator, "_difficulty_response_note", None)
                if callable(_diff_fn):
                    difficulty_note = _diff_fn(campaign.difficulty)
                    if difficulty_note:
                        response_style_note = f"{response_style_note}\n{difficulty_note}"

        # ── User prompt (mirrors engine's field ordering) ──────────────
        user_prompt = (
            f"CAMPAIGN: {campaign.name}\n"
            f"PLAYER_ID: {player.user_id}\n"
            f"IS_NEW_PLAYER: false\n"
            f"TURN_VISIBILITY_DEFAULT: {visibility_default}\n"
            f"GUARDRAILS_ENABLED: {str(campaign.guardrails).lower()}\n"
            f"RAILS_CONTEXT: {_dump_json(rails_context)}\n"
        )

        # Source material comes before game_time in new engine ordering
        if source_payload.get("available"):
            user_prompt += (
                f"SOURCE_MATERIAL_DOCS: {_dump_json(source_payload.get('docs') or [])}\n"
                f"SOURCE_MATERIAL_KEYS: {_dump_json(source_payload.get('keys') or [])}\n"
                f"SOURCE_MATERIAL_SNIPPET_COUNT: {source_payload.get('chunk_count')}\n"
                f"SOURCE_MATERIAL_CHUNK_COUNT: {source_payload.get('chunk_count')}\n"
            )
            # Source material digests (engine renders these when available)
            source_digests = source_payload.get("digests") or {}
            for digest_key, digest_text in source_digests.items():
                user_prompt += f"SOURCE_MATERIAL_DIGEST [{digest_key}]:\n{digest_text}\n"

        user_prompt += (
            f"CURRENT_GAME_TIME: {_dump_json(game_time)}\n"
            f"SPEED_MULTIPLIER: {speed_mult}\n"
            f"DIFFICULTY: {difficulty}\n"
            f"ACTIVE_PLAYER_LOCATION: {_dump_json(active_location)}\n"
            f"MEMORY_LOOKUP_ENABLED: {str(memory_enabled).lower()}\n"
            f"RECENT_TURNS_LOADED: true\n"
        )

        user_prompt += (
            f"WORLD_CHARACTERS: {_dump_json(characters_prompt)}\n"
            f"PLAYER_CARD: {_dump_json(player_card)}\n"
            f"PARTY_SNAPSHOT: {_dump_json(party_snapshot)}\n"
        )

        # Literary styles (after PARTY_SNAPSHOT, before story context)
        literary_styles_text = self._literary_styles_for_prompt(
            state.campaign_state, characters_prompt,
        )
        if literary_styles_text:
            user_prompt += f"LITERARY_STYLES:\n{literary_styles_text}\n"

        # Puzzle/minigame/dice system
        puzzle_text = self._puzzle_system_for_prompt(state.campaign_state)
        if puzzle_text:
            user_prompt += f"{puzzle_text}\n"

        # Final stage includes story context, summary, state, calendar, recent turns
        if story_context:
            user_prompt += f"STORY_CONTEXT:\n{story_context}\n"
        user_prompt += (
            f"WORLD_SUMMARY: {state.summary}\n"
            f"WORLD_STATE: {_dump_json(model_state)}\n"
            f"CALENDAR: {_dump_json(state.campaign_state.get('calendar', []))}\n"
            f"CALENDAR_REMINDERS:\n\n"
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

        # Prompt tail: mirrors _build_turn_prompt_tail() —
        # action first, then response_style_note, then WRITING_CRAFT last
        active_name = str(player_state_prompt.get("character_name") or "").strip()
        action_label = f"PLAYER_ACTION ({active_name.upper()})" if active_name else "PLAYER_ACTION"
        user_prompt += f"{action_label}: {turn.action}\n"
        if response_style_note:
            user_prompt += f"{response_style_note}\n"

        # WRITING_CRAFT guidance — in the engine this is appended during the
        # ready_to_write finalization transition. For single-shot benchmark
        # flow, append it at the tail so the model reads it last.
        writing_craft = getattr(ZorkEmulator, "WRITING_CRAFT_PROMPT", "")
        if writing_craft:
            user_prompt += writing_craft

        # ── System prompt (final-stage assembly) ───────────────────────
        # In the engine's three-stage flow, the final stage system prompt
        # contains ONLY SYSTEM_PROMPT + guardrails + on_rails. Tool prompts
        # (memory, SMS, timer, calendar, roster) belong to the bootstrap
        # and research stages. The benchmark mirrors the final stage exactly.
        system_prompt = ZorkEmulator.SYSTEM_PROMPT

        if campaign.guardrails:
            system_prompt += ZorkEmulator.GUARDRAILS_SYSTEM_PROMPT

        if campaign.on_rails:
            on_rails_prompt = getattr(ZorkEmulator, "ON_RAILS_SYSTEM_PROMPT", "")
            if on_rails_prompt:
                system_prompt += on_rails_prompt

        return system_prompt, user_prompt

    @staticmethod
    def _build_story_context(
        campaign_state: dict[str, Any],
        on_rails: bool,
    ) -> str | None:
        """Build story context from chapters in campaign state.

        Mirrors ZorkEmulator._build_story_context() which now renders
        chapter/scene structure when chapters are present in state.
        """
        if on_rails:
            # On-rails uses story_outline, not chapters
            outline = campaign_state.get("story_outline")
            if not isinstance(outline, dict):
                return None
            return _dump_json(outline)

        chapters = campaign_state.get("chapters")
        if isinstance(chapters, list) and chapters:
            active_rows = [
                row for row in chapters
                if isinstance(row, dict)
                and str(row.get("status") or "active").strip().lower() == "active"
            ]
            rows = active_rows[:4] if active_rows else [
                row for row in chapters if isinstance(row, dict)
            ][:4]
            if rows:
                current = rows[0]
                lines: list[str] = []
                lines.append(f"CURRENT CHAPTER: {current.get('title', 'Untitled')}")
                lines.append(f"  Summary: {current.get('summary', '')}")
                scenes = current.get("scenes") or []
                current_scene_slug = str(current.get("current_scene") or "").strip()
                if isinstance(scenes, list):
                    for i, scene in enumerate(scenes):
                        scene_slug = str(scene or "").strip()
                        marker = " >>> CURRENT SCENE <<<" if scene_slug and scene_slug == current_scene_slug else ""
                        label = scene_slug.replace("_", "-")
                        parts = [p for p in label.split("-") if p]
                        label = " ".join(p.capitalize() for p in parts)[:120] if parts else "Untitled"
                        lines.append(f"  Scene {i + 1}: {label}{marker}")
                if len(rows) > 1:
                    lines.append("")
                    for idx, row in enumerate(rows[1:4], start=1):
                        heading = "NEXT CHAPTER" if idx == 1 else f"UPCOMING CHAPTER {idx}"
                        lines.append(f"{heading}: {row.get('title', 'Untitled')}")
                        summary = str(row.get("summary") or "").strip()
                        if summary:
                            lines.append(f"  Preview: {summary[:320]}")
                while lines and not lines[-1]:
                    lines.pop()
                return "\n".join(lines) if lines else None

        outline = campaign_state.get("story_outline")
        if isinstance(outline, dict):
            return _dump_json(outline)

        return None

    # ── Character filtering (mirrors engine) ──────────────────────

    MAX_CHARACTERS_IN_PROMPT = 20
    MAX_CHARACTERS_CHARS = 8000

    @classmethod
    def _build_characters_for_prompt(
        cls,
        characters: dict[str, dict[str, Any]],
        player_state: dict[str, Any],
        recent_turns: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Filter and rank characters for prompt inclusion.

        Mirrors ZorkEmulator._build_characters_for_prompt():
        - Nearby characters (same location) get full data
        - Recently mentioned characters get partial data
        - Distant characters get minimal data
        """
        if not characters:
            return []

        player_location = str(player_state.get("location") or "").strip().lower()

        # Build recent text for mention detection
        recent_lower = ""
        if recent_turns:
            parts = []
            for rt in recent_turns:
                parts.append(rt.get("action", ""))
                parts.append(rt.get("narration", ""))
            recent_lower = " ".join(parts).lower()

        nearby: list[dict[str, Any]] = []
        mentioned: list[dict[str, Any]] = []
        distant: list[dict[str, Any]] = []

        for slug, char in characters.items():
            if not isinstance(char, dict):
                continue
            char_location = str(char.get("location") or "").strip().lower()
            char_name = str(char.get("name") or slug).strip().lower()
            is_deceased = bool(char.get("deceased_reason"))

            if not is_deceased and player_location and char_location == player_location:
                entry = dict(char)
                entry["_slug"] = slug
                nearby.append(entry)
            elif char_name in recent_lower or slug in recent_lower:
                entry = {
                    "_slug": slug,
                    "name": char.get("name", slug),
                    "speech_style": char.get("speech_style"),
                    "literary_style": char.get("literary_style"),
                    "location": char.get("location"),
                    "current_status": char.get("current_status"),
                    "allegiance": char.get("allegiance"),
                }
                if is_deceased:
                    entry["deceased_reason"] = char.get("deceased_reason")
                mentioned.append(entry)
            else:
                entry: dict[str, Any] = {"_slug": slug, "name": char.get("name", slug)}
                if is_deceased:
                    entry["deceased_reason"] = char.get("deceased_reason")
                else:
                    entry["location"] = char.get("location")
                distant.append(entry)

        result = nearby + mentioned + distant
        result = result[:cls.MAX_CHARACTERS_IN_PROMPT]
        return cls._fit_characters_to_budget(result, cls.MAX_CHARACTERS_CHARS)

    @classmethod
    def _fit_characters_to_budget(
        cls,
        characters_list: list[dict[str, Any]],
        max_chars: int,
    ) -> list[dict[str, Any]]:
        """Truncate character list to fit within a JSON-serialized byte budget."""
        while characters_list:
            text = json.dumps(characters_list, ensure_ascii=True)
            if len(text) <= max_chars:
                return characters_list
            characters_list = characters_list[:-1]
        return []

    # ── Literary styles ──────────────────────────────────────────

    MAX_LITERARY_STYLES_PROMPT_CHARS = 3000

    @classmethod
    def _literary_styles_for_prompt(
        cls,
        campaign_state: dict[str, Any],
        characters_for_prompt: list[dict[str, Any]],
    ) -> str | None:
        """Render literary style profiles for characters in prompt.

        Mirrors ZorkEmulator._literary_styles_for_prompt().
        """
        styles = campaign_state.get("literary_styles")
        if not isinstance(styles, dict) or not styles:
            return None

        # Collect referenced styles from active characters
        active_refs: set[str] = set()
        for char in characters_for_prompt or []:
            if not isinstance(char, dict):
                continue
            ref = str(char.get("literary_style") or "").strip()
            if ref:
                active_refs.add(ref)

        def _sort_key(key: str) -> tuple[int, str]:
            return (0 if key in active_refs else 1, key)

        lines: list[str] = []
        budget = cls.MAX_LITERARY_STYLES_PROMPT_CHARS
        for key in sorted(styles.keys(), key=_sort_key):
            entry = styles.get(key)
            if not isinstance(entry, dict):
                continue
            profile = str(entry.get("profile") or "").strip()
            if not profile:
                continue
            line = f"  {key}: {profile}"
            if len(line) > budget:
                break
            lines.append(line)
            budget -= len(line) + 1
            if budget <= 0:
                break
        return "\n".join(lines) if lines else None

    # ── Puzzle / minigame / dice system ──────────────────────────

    @staticmethod
    def _puzzle_system_for_prompt(campaign_state: dict[str, Any]) -> str | None:
        """Render active puzzle, minigame, and dice check state for prompt.

        Mirrors ZorkEmulator._puzzle_system_for_prompt().
        """
        parts: list[str] = []

        puzzle_mode = campaign_state.get("puzzle_mode")
        if puzzle_mode and puzzle_mode != "none":
            parts.append(f"PUZZLE_CONFIG:\n  mode: {puzzle_mode}")

        active_puzzle = campaign_state.get("_active_puzzle")
        if isinstance(active_puzzle, dict):
            lines = ["ACTIVE_PUZZLE:"]
            for k, v in active_puzzle.items():
                if not str(k).startswith("_"):
                    lines.append(f"  {k}: {v}")
            parts.append("\n".join(lines))

        puzzle_result = campaign_state.get("_puzzle_result")
        if isinstance(puzzle_result, dict):
            lines = ["PUZZLE_RESULT:"]
            for k, v in puzzle_result.items():
                lines.append(f"  {k}: {v}")
            parts.append("\n".join(lines))

        active_minigame = campaign_state.get("_active_minigame")
        if isinstance(active_minigame, dict):
            lines = ["ACTIVE_MINIGAME:"]
            for k, v in active_minigame.items():
                if not str(k).startswith("_"):
                    lines.append(f"  {k}: {v}")
            parts.append("\n".join(lines))

        minigame_result = campaign_state.get("_minigame_result")
        if isinstance(minigame_result, dict):
            lines = ["MINIGAME_RESULT:"]
            for k, v in minigame_result.items():
                lines.append(f"  {k}: {v}")
            parts.append("\n".join(lines))

        last_dice = campaign_state.get("_last_dice_check")
        if isinstance(last_dice, dict):
            attr = last_dice.get("attribute", "skill")
            roll_val = last_dice.get("roll", 0)
            mod = last_dice.get("modifier", 0)
            total = last_dice.get("total", 0)
            dc = last_dice.get("dc", 0)
            success = last_dice.get("success", False)
            context = last_dice.get("context", "")
            parts.append(
                f"LAST_DICE_CHECK:\n"
                f"  attribute: {attr}\n"
                f"  roll: {roll_val} + {mod} = {total} vs DC {dc}\n"
                f"  result: {'success' if success else 'failure'}\n"
                f'  context: "{context}"'
            )

        return "\n\n".join(parts) if parts else None

    # ── Memory lookup ────────────────────────────────────────────

    MEMORY_LOOKUP_MIN_SUMMARY_CHARS = 2000
    _MEMORY_LOOKUP_MARKERS = (
        "remember", "recall", "what happened", "previously",
        "backstory", "history", "who is", "what is",
        "according to", "from the book", "from source",
        "source material", "canon", "lore", "look up",
    )

    @classmethod
    def _memory_lookup_enabled(
        cls,
        campaign_memory_flag: bool,
        summary: str,
        action: str,
        *,
        source_available: bool = False,
    ) -> bool:
        """Determine if memory lookup should be enabled for this turn.

        Mirrors ZorkEmulator._memory_lookup_enabled_for_prompt():
        - True if summary is long enough to warrant memory search
        - True if source material available and action requests lookup
        - False otherwise (even if campaign has memory enabled)
        """
        if not campaign_memory_flag:
            return False

        summary_len = len(str(summary or "").strip())
        if summary_len >= cls.MEMORY_LOOKUP_MIN_SUMMARY_CHARS:
            return True

        if source_available:
            text = " ".join(str(action or "").strip().lower().split())
            if text and not re.match(r"\s*\[ooc\b", text, re.IGNORECASE):
                if any(marker in text for marker in cls._MEMORY_LOOKUP_MARKERS):
                    return True

        return False
