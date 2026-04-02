"""Tests for prompt_builder.py — AccumulatedState, character filtering, literary styles, etc."""

import json

from tgb.config import Scenario, CampaignSetup, PlayerSetup, TurnSpec
from tgb.prompt_builder import AccumulatedState, PromptBuilder


def _make_scenario(**kwargs) -> Scenario:
    return Scenario(
        name=kwargs.get("name", "test"),
        description="test",
        tags=[],
        tier="basic",
        campaign=kwargs.get("campaign", CampaignSetup(name="test")),
        player=kwargs.get("player", PlayerSetup()),
        turns=[TurnSpec(action="test")],
        recent_turns=kwargs.get("recent_turns", []),
    )


def _make_state(scenario=None, **overrides) -> AccumulatedState:
    scenario = scenario or _make_scenario()
    state = AccumulatedState(scenario)
    for k, v in overrides.items():
        setattr(state, k, v)
    return state


# ── AccumulatedState.apply() tracking ────────────────────────────

class TestAccumulatedStateCalendar:
    def test_calendar_add(self):
        state = _make_state()
        state.apply({
            "calendar_update": {
                "add": [{"name": "festival", "time_remaining": 3, "time_unit": "days"}],
            },
        })
        calendar = state.campaign_state.get("calendar", [])
        assert len(calendar) == 1
        assert calendar[0]["name"] == "festival"

    def test_calendar_remove(self):
        state = _make_state()
        state.campaign_state["calendar"] = [
            {"name": "old-event", "time_remaining": 1, "time_unit": "hours"},
            {"name": "keep-me", "time_remaining": 5, "time_unit": "days"},
        ]
        state.apply({
            "calendar_update": {
                "remove": ["old-event"],
            },
        })
        calendar = state.campaign_state.get("calendar", [])
        assert len(calendar) == 1
        assert calendar[0]["name"] == "keep-me"

    def test_calendar_add_and_remove(self):
        state = _make_state()
        state.campaign_state["calendar"] = [
            {"name": "expire-me", "time_remaining": 0, "time_unit": "hours"},
        ]
        state.apply({
            "calendar_update": {
                "add": [{"name": "new-event", "time_remaining": 2, "time_unit": "hours"}],
                "remove": ["expire-me"],
            },
        })
        calendar = state.campaign_state["calendar"]
        assert len(calendar) == 1
        assert calendar[0]["name"] == "new-event"


class TestCalendarRenderedInPrompt:
    def test_calendar_data_in_prompt(self):
        """CALENDAR in prompt should contain tracked calendar events."""
        scenario = _make_scenario(
            campaign=CampaignSetup(name="test"),
        )
        state = AccumulatedState(scenario)
        state.apply({
            "calendar_update": {
                "add": [{"name": "festival", "time_remaining": 3, "time_unit": "days"}],
            },
        })
        builder = PromptBuilder()
        _, user_prompt = builder.build(scenario, TurnSpec(action="look"), state)
        # The calendar line should contain the tracked event, not {}
        assert "festival" in user_prompt
        assert 'CALENDAR: []' not in user_prompt


class TestAccumulatedStateGiveItem:
    def test_give_item_removes_from_inventory(self):
        state = _make_state()
        state.player_state["inventory"] = ["sword", "shield", "potion"]
        state.apply({
            "give_item": {"item": "sword", "to_actor_id": "123"},
        })
        assert "sword" not in state.player_state["inventory"]
        assert "shield" in state.player_state["inventory"]

    def test_give_item_case_insensitive(self):
        state = _make_state()
        state.player_state["inventory"] = ["Golden Key", "torch"]
        state.apply({
            "give_item": {"item": "golden key", "to_discord_mention": "<@456>"},
        })
        assert len(state.player_state["inventory"]) == 1
        assert state.player_state["inventory"][0] == "torch"


class TestAccumulatedStateTimer:
    def test_timer_tracked(self):
        state = _make_state()
        state.apply({
            "set_timer_delay": 60,
            "set_timer_event": "The guard returns",
            "set_timer_interruptible": False,
        })
        timer = state.campaign_state.get("_active_timer")
        assert timer is not None
        assert timer["delay"] == 60
        assert timer["event"] == "The guard returns"
        assert timer["interruptible"] is False


class TestAccumulatedStateDiceCheck:
    def test_dice_check_tracked(self):
        state = _make_state()
        state.apply({
            "dice_check": {"attribute": "strength", "dc": 15, "context": "lift boulder"},
        })
        assert state.campaign_state["_last_dice_check"]["attribute"] == "strength"

    def test_puzzle_trigger_tracked(self):
        state = _make_state()
        state.apply({
            "puzzle_trigger": {"puzzle_type": "riddle", "context": "sphinx", "difficulty": "hard"},
        })
        assert state.campaign_state["_active_puzzle"]["puzzle_type"] == "riddle"

    def test_minigame_challenge_tracked(self):
        state = _make_state()
        state.apply({
            "minigame_challenge": {"game_type": "tic-tac-toe", "opponent_slug": "tavern-keeper"},
        })
        assert state.campaign_state["_active_minigame"]["game_type"] == "tic-tac-toe"


# ── Character filtering ─────────────────────────────────────────

class TestBuildCharactersForPrompt:
    def test_empty_characters(self):
        result = PromptBuilder._build_characters_for_prompt({}, {}, [])
        assert result == []

    def test_nearby_characters_first(self):
        chars = {
            "guard": {"name": "Guard", "location": "gate"},
            "wizard": {"name": "Wizard", "location": "tower"},
        }
        player_state = {"location": "gate"}
        result = PromptBuilder._build_characters_for_prompt(chars, player_state, [])
        assert result[0]["_slug"] == "guard"

    def test_mentioned_characters_before_distant(self):
        chars = {
            "guard": {"name": "Guard", "location": "gate"},
            "alice": {"name": "Alice", "location": "forest"},
        }
        player_state = {"location": "town"}
        recent = [{"action": "ask about Alice", "narration": "You think of Alice."}]
        result = PromptBuilder._build_characters_for_prompt(chars, player_state, recent)
        slugs = [c["_slug"] for c in result]
        assert slugs.index("alice") < slugs.index("guard")

    def test_deceased_not_nearby(self):
        chars = {
            "ghost": {"name": "Ghost", "location": "gate", "deceased_reason": "killed"},
        }
        player_state = {"location": "gate"}
        result = PromptBuilder._build_characters_for_prompt(chars, player_state, [])
        # Deceased shouldn't be in "nearby" tier — should be distant with death info
        assert result[0].get("deceased_reason") == "killed"
        # Distant characters don't get full data
        assert "personality" not in result[0]

    def test_max_characters_limit(self):
        chars = {f"npc-{i}": {"name": f"NPC {i}"} for i in range(30)}
        result = PromptBuilder._build_characters_for_prompt(chars, {}, [])
        assert len(result) <= PromptBuilder.MAX_CHARACTERS_IN_PROMPT

    def test_budget_truncation(self):
        # Create characters with very long data to test budget
        chars = {
            f"npc-{i}": {"name": f"NPC {i}", "background": "x" * 2000}
            for i in range(10)
        }
        player_state = {"location": "here"}
        # All at same location so they get full data
        for c in chars.values():
            c["location"] = "here"
        result = PromptBuilder._build_characters_for_prompt(chars, player_state, [])
        serialized = json.dumps(result, ensure_ascii=True)
        assert len(serialized) <= PromptBuilder.MAX_CHARACTERS_CHARS


# ── Literary styles ──────────────────────────────────────────────

class TestLiteraryStylesForPrompt:
    def test_no_styles(self):
        result = PromptBuilder._literary_styles_for_prompt({}, [])
        assert result is None

    def test_renders_styles(self):
        campaign_state = {
            "literary_styles": {
                "noir": {"profile": "Dark, moody prose with clipped sentences."},
                "epic": {"profile": "Grand sweeping descriptions."},
            },
        }
        chars = [{"_slug": "detective", "literary_style": "noir"}]
        result = PromptBuilder._literary_styles_for_prompt(campaign_state, chars)
        assert result is not None
        assert "noir" in result
        assert "epic" in result

    def test_active_refs_sorted_first(self):
        campaign_state = {
            "literary_styles": {
                "aaa": {"profile": "A style"},
                "zzz": {"profile": "Z style"},
            },
        }
        chars = [{"literary_style": "zzz"}]
        result = PromptBuilder._literary_styles_for_prompt(campaign_state, chars)
        lines = result.split("\n")
        # zzz should come before aaa because it's actively referenced
        assert "zzz" in lines[0]


# ── Puzzle system ────────────────────────────────────────────────

class TestPuzzleSystemForPrompt:
    def test_no_puzzle(self):
        result = PromptBuilder._puzzle_system_for_prompt({})
        assert result is None

    def test_puzzle_mode(self):
        result = PromptBuilder._puzzle_system_for_prompt({"puzzle_mode": "riddle"})
        assert "PUZZLE_CONFIG" in result
        assert "riddle" in result

    def test_active_puzzle(self):
        result = PromptBuilder._puzzle_system_for_prompt({
            "_active_puzzle": {"type": "riddle", "text": "What has keys?"},
        })
        assert "ACTIVE_PUZZLE" in result

    def test_dice_check(self):
        result = PromptBuilder._puzzle_system_for_prompt({
            "_last_dice_check": {
                "attribute": "strength", "roll": 15, "modifier": 3,
                "total": 18, "dc": 12, "success": True, "context": "lift gate",
            },
        })
        assert "LAST_DICE_CHECK" in result
        assert "success" in result
        assert "strength" in result

    def test_minigame(self):
        result = PromptBuilder._puzzle_system_for_prompt({
            "_active_minigame": {"game_type": "tic-tac-toe", "status": "player_turn"},
        })
        assert "ACTIVE_MINIGAME" in result


# ── Memory lookup ────────────────────────────────────────────────

class TestMemoryLookupEnabled:
    def test_disabled_when_no_flag(self):
        assert not PromptBuilder._memory_lookup_enabled(False, "", "look around")

    def test_enabled_with_long_summary(self):
        assert PromptBuilder._memory_lookup_enabled(True, "x" * 2100, "look around")

    def test_disabled_with_short_summary(self):
        assert not PromptBuilder._memory_lookup_enabled(True, "short", "look around")

    def test_enabled_with_source_and_marker(self):
        assert PromptBuilder._memory_lookup_enabled(
            True, "short", "what happened yesterday", source_available=True,
        )

    def test_disabled_with_source_but_no_marker(self):
        assert not PromptBuilder._memory_lookup_enabled(
            True, "short", "look around", source_available=True,
        )

    def test_disabled_ooc_with_marker(self):
        assert not PromptBuilder._memory_lookup_enabled(
            True, "short", "[OOC] remember to save", source_available=True,
        )


# ── Model state exclude keys ────────────────────────────────────

class TestModelStateExclusion:
    def test_excludes_internal_keys(self):
        """The prompt builder should exclude engine-internal state keys."""
        scenario = _make_scenario(
            campaign=CampaignSetup(
                name="test",
                state={
                    "tone": "dark",
                    "game_time": {"day": 1},
                    "_active_puzzle": {"type": "riddle"},
                    "_sms_threads": [],
                    "literary_styles": {},
                    "story_outline": {"act1": "begin"},
                    "room_id": "cave",
                },
            ),
        )
        state = AccumulatedState(scenario)
        builder = PromptBuilder()
        _, user_prompt = builder.build(scenario, TurnSpec(action="look"), state)
        # Internal keys should NOT appear in WORLD_STATE
        assert "_active_puzzle" not in user_prompt.split("WORLD_STATE:")[1].split("\n")[0]
        assert "_sms_threads" not in user_prompt.split("WORLD_STATE:")[1].split("\n")[0]
        # But visible keys should
        assert "tone" in user_prompt


# ── Prompt tail ordering ────────────────────────────────────────

# ── WRITING_CRAFT prompt ──────────────────────────────────────

class TestWritingCraftInPrompt:
    def test_writing_craft_present(self):
        """WRITING_CRAFT section should appear in the user prompt."""
        scenario = _make_scenario(campaign=CampaignSetup(name="test"))
        state = AccumulatedState(scenario)
        builder = PromptBuilder()
        _, user_prompt = builder.build(scenario, TurnSpec(action="look"), state)
        assert "WRITING_CRAFT:" in user_prompt

    def test_writing_craft_after_action(self):
        """WRITING_CRAFT should come after PLAYER_ACTION."""
        scenario = _make_scenario(campaign=CampaignSetup(name="test"))
        state = AccumulatedState(scenario)
        builder = PromptBuilder()
        _, user_prompt = builder.build(scenario, TurnSpec(action="look"), state)
        action_pos = user_prompt.find("PLAYER_ACTION")
        craft_pos = user_prompt.find("WRITING_CRAFT:")
        assert action_pos >= 0
        assert craft_pos > action_pos

    def test_writing_craft_contains_key_directives(self):
        """WRITING_CRAFT should contain the engine's craft principles."""
        scenario = _make_scenario(campaign=CampaignSetup(name="test"))
        state = AccumulatedState(scenario)
        builder = PromptBuilder()
        _, user_prompt = builder.build(scenario, TurnSpec(action="look"), state)
        assert "concrete" in user_prompt.lower() or "sensory" in user_prompt.lower()
        assert "precise word" in user_prompt.lower() or "vivid verb" in user_prompt.lower()


# ── Prompt tail ordering ────────────────────────────────────────

class TestPromptTailOrdering:
    def test_action_before_style_note(self):
        """PLAYER_ACTION should come before response_style_note in the tail."""
        scenario = _make_scenario(
            campaign=CampaignSetup(name="test"),
            player=PlayerSetup(state={"character_name": "Alice"}),
        )
        state = AccumulatedState(scenario)
        builder = PromptBuilder()
        _, user_prompt = builder.build(scenario, TurnSpec(action="look around"), state)
        action_pos = user_prompt.find("PLAYER_ACTION")
        assert action_pos >= 0
        # Style note (if present) should be after the action
        style_markers = ["RESPONSE_STYLE", "narration style", "Be concise"]
        for marker in style_markers:
            marker_pos = user_prompt.find(marker)
            if marker_pos >= 0:
                assert marker_pos > action_pos, (
                    f"Style marker '{marker}' at {marker_pos} should be after "
                    f"PLAYER_ACTION at {action_pos}"
                )
