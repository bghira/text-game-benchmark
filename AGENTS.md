# text-game-benchmark

Benchmark harness for text-game-engine that tests LLM compliance with the engine's structured output contract.

## Architecture

- `src/tgb/` — main package
  - `cli.py` — argparse entry point (`tgb` command)
  - `config.py` — YAML scenario loader into frozen dataclasses
  - `prompt_builder.py` — builds (system_prompt, user_prompt) from scenario YAML
  - `response_parser.py` — JSON extraction from raw model output
  - `runner.py` — orchestration: scenario → model → judge → results
  - `results.py` — JSON result serialization
  - `judge.py` — judge model evaluator
  - `clients/` — model backends (ollama wrapper, OpenAI-compatible)
  - `checks/` — pure-function check implementations + registry
- `scenarios/` — YAML scenario definitions
- `results/` — output directory (gitignored)
- `tests/` — unit tests

## Key Invariants

- Scenarios are pure YAML — no Python code in scenario files
- Checks are pure functions: `(parsed, scenario, turn, state) → CheckResult`
- Judge is a separate client call, skippable with `--no-judge`
- We import constants from text-game-engine but never its persistence layer

## Adding a Check

1. Write a pure function in the appropriate `checks/*.py` module
2. Return `CheckResult(check_id, passed, detail, category)`
3. Register in `checks/registry.py` CHECKS dict
4. Reference by `check_id` in scenario YAML

## Adding a Scenario

1. Copy `scenarios/_template.yaml`
2. Fill in campaign, player, turns, and checks
3. Run: `tgb -s scenarios/your_scenario.yaml -m ollama:qwen2.5:32b`

## Running

```
tgb -s scenarios/alice_basic.yaml -m ollama:qwen2.5:32b
tgb -s scenarios/ -m ollama:qwen2.5:32b --no-judge
tgb --list-checks
```
